"""Training and evaluation entry points for adaptive and shallow baselines."""
from __future__ import annotations
import argparse, json, math, os, random, subprocess, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from transformers import AutoTokenizer
from .data import AspectDataset, load_records, SentenceCollator
from .model import AdaptiveDepthModel
from .losses import adaptive_loss, gaussian_nll
from .metrics import regression_metrics, unnormalize
from .infer import simulate_exit, simulate_variance_exit


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch for repeatable runs."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def select_device() -> torch.device:
    """Choose CUDA, Apple MPS, or CPU in that order."""
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


class SentenceBatchSampler(Sampler[list[int]]):
    """Pack whole sentence groups into batches of at most N unique sentences."""
    def __init__(self, examples, batch_size=32, shuffle=False):
        self.groups=defaultdict(list)
        for i,ex in enumerate(examples): self.groups[(ex.example_id,ex.text)].append(i)
        self.batch_size,self.shuffle=batch_size,shuffle
    def __iter__(self):
        groups=list(self.groups.values())
        if self.shuffle: random.shuffle(groups)
        batch=[]; n=0
        for group in groups:
            if batch and n>=self.batch_size: yield batch; batch=[]; n=0
            batch.extend(group); n+=1
        if batch: yield batch
    def __len__(self): return math.ceil(len(self.groups)/self.batch_size)


def make_loader(examples, tokenizer, batch_size=32, shuffle=False):
    return DataLoader(AspectDataset(examples), batch_sampler=SentenceBatchSampler(examples,batch_size,shuffle),
                      collate_fn=SentenceCollator(tokenizer), num_workers=0)


def _git_commit(root: Path) -> str:
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return "unavailable"


def _to_device(batch, device):
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, threshold=.5, fixed_layer=None, variance_threshold=None):
    """Evaluate final-layer and threshold predictions in original units."""
    model.eval(); ys=[]; last=[]; exited=[]; depths=[]; sentence_depths=[]
    for batch in loader:
        batch=_to_device(batch,device); target=batch["targets"].cpu().numpy()
        mu,lv,lam,_=model(batch)
        if fixed_layer is not None:
            k=max(1,min(int(fixed_layer),mu.shape[0])); p=mu[k-1]
            d=torch.full((mu.shape[1],),k,device=mu.device,dtype=torch.long)
        elif variance_threshold is not None:
            p,d=simulate_variance_exit(mu,lv,variance_threshold)
        else: p,d=simulate_exit(mu,lam,threshold)
        sd=[int(d[batch["sentence_index"]==i].max()) for i in range(len(batch["texts"]))]
        ys.append(target); last.append(unnormalize(mu[-1]).cpu().numpy()); exited.append(unnormalize(p).cpu().numpy())
        depths.append(d.cpu().numpy()); sentence_depths.extend(sd)
    return np.concatenate(ys),np.concatenate(last),np.concatenate(exited),np.concatenate(depths),np.asarray(sentence_depths)


def run(args):
    root=Path(args.root).resolve(); data_dir=root/"data"
    all_files=[p for p in data_dir.rglob("*") if p.suffix.lower() in {".json",".jsonl"}]
    if args.quick: all_files=[p for p in all_files if "synthetic" in p.parts]
    else: all_files=[p for p in all_files if "synthetic" not in p.parts]
    train_files=sorted(p for p in all_files if "dev" not in p.stem.lower() and "valid" not in p.stem.lower() and "test" not in p.stem.lower())
    dev_files=sorted(p for p in all_files if "dev" in p.stem.lower() or "valid" in p.stem.lower())
    if not train_files: raise FileNotFoundError(f"No train JSON/JSONL found under {data_dir}. Run scripts/inspect_data.py after copying data.")
    train=[x for p in train_files for x in load_records(p)]
    rng=np.random.default_rng(args.seed)
    ids=sorted({x.example_id for x in train}); order=rng.permutation(len(ids)); n=max(1,int(.1*len(order)))
    valid_ids={ids[i] for i in order[:n]}
    valid=[x for x in train if x.example_id in valid_ids]; train=[x for x in train if x.example_id not in valid_ids]
    if dev_files:
        eval_examples=[x for p in dev_files for x in load_records(p)]
        split_note="seeded 10% sentence-level training holdout for validation; official dev files used for evaluation"
    else:
        eval_examples=valid
        split_note="seeded 10% sentence-level training holdout used for validation and evaluation; hidden test was not used"
    if args.quick: train=train[:max(32,min(128,len(train)))]; valid=valid[:max(16,min(64,len(valid)))]; args.epochs=1
    if args.max_train_sentences:
        sentence_ids=sorted({x.example_id for x in train})
        rng=np.random.default_rng(args.seed); rng.shuffle(sentence_ids)
        keep=set(sentence_ids[:args.max_train_sentences])
        train=[x for x in train if x.example_id in keep]
    seed_everything(args.seed); device=select_device()
    print(f"device={device}; train_sentences={len({x.example_id for x in train})}; validation_sentences={len({x.example_id for x in valid})}; eval_sentences={len({x.example_id for x in eval_examples})}; epochs={args.epochs}",flush=True)
    print("Loading tokenizer and encoder...",flush=True)
    tokenizer=AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    if args.method=="B1_pair_input":
        from .pair_baseline import run_pair
        run_pair(args,root,train,valid,eval_examples,tokenizer,device,split_note)
        return
    model=AdaptiveDepthModel(args.encoder,gate_detach_inputs=not args.gate_train_inputs,
                             gate_use_variance=not args.no_gate_variance).to(device)
    train_loader=make_loader(train,tokenizer,args.batch_size,True); valid_loader=make_loader(valid,tokenizer,args.batch_size)
    eval_loader=make_loader(eval_examples,tokenizer,args.batch_size)
    updates=max(1,len(train_loader)*args.epochs); warmup=int(.1*updates); step=0
    enc_params=list(model.encoder.parameters()); head_params=[p for n,p in model.named_parameters() if not n.startswith("encoder.")]
    optim=torch.optim.AdamW([{"params":enc_params,"lr":2e-5},{"params":head_params,"lr":1e-3}])
    scaler=torch.amp.GradScaler("cuda",enabled=args.amp and device.type=="cuda")
    history=[]; best_rmse=float("inf"); best_state=None
    stage1_end=max(1,args.epochs//2) if args.two_stage else 0
    for epoch in range(args.epochs):
        stage=("stage1" if epoch<stage1_end else "stage2") if args.two_stage else "joint"
        if args.two_stage and epoch>=stage1_end:
            for p in model.encoder.parameters(): p.requires_grad_(False)
            for p in model.readout.parameters(): p.requires_grad_(False)
            for p in model.heads.parameters(): p.requires_grad_(False)
        else:
            for p in model.parameters(): p.requires_grad_(True)
        model.train()
        if stage=="stage2" and args.two_stage:
            model.encoder.eval(); model.readout.eval(); model.heads.eval()
        for batch_idx,batch in enumerate(train_loader,1):
            step+=1; batch=_to_device(batch,device); target=(batch["targets"]-5)/4
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=args.amp and device.type=="cuda"):
                mu,lv,lam,_=model(batch)
                if stage=="stage1":
                    loss,logs=adaptive_loss(mu,lv,target,lam,args.alpha,args.beta,"stage1")
                elif args.method=="B2_last":
                    loss=gaussian_nll(mu[-1],lv[-1],target).mean()
                elif args.method in {"deep_supervised","B3_shallow","B4_variance"}:
                    layer_losses=torch.stack([gaussian_nll(mu[i],lv[i],target) for i in range(mu.shape[0])])
                    loss=layer_losses.mean()
                else:
                    loss,logs=adaptive_loss(mu,lv,target,lam,args.alpha,args.beta,"joint")
            if step<=warmup: factor=step/max(1,warmup)
            else: factor=max(0.,(updates-step+1)/max(1,updates-warmup))
            for group,base in zip(optim.param_groups,[2e-5,1e-3]): group["lr"]=base*factor
            scaler.scale(loss).backward(); scaler.unscale_(optim); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optim); scaler.update()
            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                print(f"epoch={epoch+1}/{args.epochs} batch={batch_idx}/{len(train_loader)} loss={float(loss.detach()):.4f}",flush=True)
        y,last,exited,d,sd=evaluate(model,valid_loader,device,.5); metrics=regression_metrics(last,y)
        exit_metrics=regression_metrics(exited,y)
        history.append({"epoch":epoch+1,"stage":stage,**metrics,"exit_RMSE_VA":exit_metrics["RMSE_VA"],
                        "mean_exit_layer":float(np.mean(d)),"mean_sentence_layer":float(np.mean(sd))})
        if metrics["RMSE_VA"] <= best_rmse:
            best_rmse=metrics["RMSE_VA"]
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"epoch={epoch+1} validation={json.dumps(history[-1])}")
    if best_state is not None: model.load_state_dict(best_state)
    result_dir=root/("results_smoke" if args.quick else "results"); result_dir.mkdir(exist_ok=True)
    ckpt_dir=root/("results_smoke/checkpoints" if args.quick else "checkpoints"); ckpt_dir.mkdir(parents=True,exist_ok=True)
    name=f"{args.method}_seed{args.seed}"; ckpt=ckpt_dir/f"{name}.pt"
    torch.save({"state_dict":model.state_dict(),"encoder":args.encoder,"seed":args.seed},ckpt)
    if args.method=="B3_shallow":
        y,last,p,d,sd=evaluate(model,eval_loader,device,args.tau,fixed_layer=args.fixed_layer)
    elif args.method=="B4_variance":
        y,last,p,d,sd=evaluate(model,eval_loader,device,args.tau,variance_threshold=args.theta)
    else:
        y,last,p,d,sd=evaluate(model,eval_loader,device,args.tau)
    if args.method=="B2_last" or args.method=="deep_supervised":
        p=last; d=np.full(len(y),model.layers,dtype=int); sd=np.full(len(eval_examples),model.layers,dtype=int)
    metrics=regression_metrics(p,y)
    row={"method":args.method,"seed":args.seed,"tau":args.tau,"alpha":args.alpha,"beta":args.beta,
         "split":split_note,"avg_aspect_layers":float(d.mean()),"avg_sentence_layers":float(sd.mean()),
         "FLOPs":None,"latency_ms_median":None,"latency_ms_p95":None,"device":str(device),**metrics}
    (ckpt_dir/f"{name}.json").write_text(json.dumps({"config":vars(args),"git_commit":_git_commit(root),
                                                     "history":history,"metrics":row},indent=2),encoding="utf-8")
    import csv
    out=result_dir/"runs.csv"; exists=out.exists()
    with out.open("a",newline="",encoding="utf-8") as f:
        if exists:
            with out.open(newline="",encoding="utf-8") as existing_file: fields=next(csv.reader(existing_file))
        else: fields=list(row)
        w=csv.DictWriter(f,fieldnames=fields)
        if not exists: w.writeheader()
        w.writerow(row)
    print(f"saved {ckpt} and {out}")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--encoder",default="bert-base-uncased"); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=3); p.add_argument("--batch_size",type=int,default=32)
    p.add_argument("--alpha",type=float,default=.3); p.add_argument("--beta",type=float,default=.003)
    p.add_argument("--tau",type=float,default=.5); p.add_argument("--method",default="ours")
    p.add_argument("--fixed_layer",type=int,default=6); p.add_argument("--theta",type=float,default=1.0)
    p.add_argument("--two_stage",action="store_true"); p.add_argument("--gate_train_inputs",action="store_true")
    p.add_argument("--no_gate_variance",action="store_true")
    p.add_argument("--amp",action="store_true"); p.add_argument("--quick",action="store_true")
    p.add_argument("--max_train_sentences",type=int,default=0,help="Optional random sentence cap for a real-data pilot; 0 uses all training sentences.")
    run(p.parse_args())
if __name__=="__main__": main()

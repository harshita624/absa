"""B1 pair-input baseline: re-encode sentence/aspect pairs independently."""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel
from .data import AspectDataset
from .losses import gaussian_nll
from .metrics import regression_metrics
from .cost import analytic_flops


class PairInputRegressor(nn.Module):
    """Transformer regression model over [CLS] sentence [SEP] aspect [SEP]."""
    def __init__(self, encoder_name: str):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(encoder_name)
        d=self.encoder.config.hidden_size
        self.head=nn.Sequential(nn.Dropout(.1),nn.Linear(d,d),nn.GELU(),nn.Linear(d,4))
    def forward(self,batch):
        out=self.encoder(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],
                         token_type_ids=batch.get("token_type_ids"),return_dict=True)
        raw=self.head(out.last_hidden_state[:,0])
        return raw[:,:2],raw[:,2:].clamp(-6,4)


class PairCollator:
    """Tokenize each sentence and aspect as a separate transformer pair."""
    def __init__(self,tokenizer,max_length=256): self.tokenizer,self.max_length=tokenizer,max_length
    def __call__(self,examples):
        enc=self.tokenizer([x.text for x in examples],[x.aspect for x in examples],padding=True,truncation=True,
                           max_length=self.max_length,return_tensors="pt")
        enc["targets"]=torch.tensor([[x.valence,x.arousal] for x in examples],dtype=torch.float32)
        return enc


def run_pair(args,root,train,valid,eval_examples,tokenizer,device,split_note):
    """Train/evaluate B1, then save a result row and checkpoint."""
    model=PairInputRegressor(args.encoder).to(device)
    train_loader=DataLoader(AspectDataset(train),batch_size=args.batch_size,shuffle=True,collate_fn=PairCollator(tokenizer))
    valid_loader=DataLoader(AspectDataset(valid),batch_size=args.batch_size,shuffle=False,collate_fn=PairCollator(tokenizer))
    dev_loader=DataLoader(AspectDataset(eval_examples),batch_size=args.batch_size,shuffle=False,collate_fn=PairCollator(tokenizer))
    enc=list(model.encoder.parameters()); head=list(model.head.parameters())
    optim=torch.optim.AdamW([{"params":enc,"lr":2e-5},{"params":head,"lr":1e-3}])
    steps=max(1,len(train_loader)*args.epochs); warmup=max(1,int(.1*steps)); step=0; best=float("inf"); best_state=None
    def predict(loader):
        model.eval(); ys=[]; ps=[]
        with torch.inference_mode():
            for batch in loader:
                batch={k:v.to(device) for k,v in batch.items()}
                mu,lv=model(batch)
                ys.append(batch["targets"].cpu().numpy()); ps.append((mu*4+5).cpu().numpy())
        return np.concatenate(ys),np.concatenate(ps)
    for ep in range(args.epochs):
        model.train()
        for batch in train_loader:
            step+=1; batch={k:v.to(device) for k,v in batch.items()}
            optim.zero_grad(set_to_none=True)
            mu,lv=model(batch); loss=gaussian_nll(mu,lv,(batch["targets"]-5)/4).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            factor=step/warmup if step<=warmup else max(0,(steps-step+1)/max(1,steps-warmup))
            for pg,base in zip(optim.param_groups,[2e-5,1e-3]): pg["lr"]=base*factor
            optim.step()
        vy,vp=predict(valid_loader); score=regression_metrics(vp,vy)["RMSE_VA"]
        if score<=best:
            best=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"epoch={ep+1}/{args.epochs} B1_validation_RMSE_VA={score:.4f}",flush=True)
    if best_state: model.load_state_dict(best_state)
    y,pred=predict(dev_loader); metrics=regression_metrics(pred,y)
    # Pair-input baseline always re-encodes each aspect at the full depth.
    latencies=[]
    model.eval()
    with torch.inference_mode():
        for ex in eval_examples[:min(10,len(eval_examples))]:
            batch=PairCollator(tokenizer)([ex]); batch={k:v.to(device) for k,v in batch.items()}; model(batch)
        for ex in eval_examples[10:]:
            b=PairCollator(tokenizer)([ex]); b={k:v.to(device) for k,v in b.items()}
            t=time.perf_counter(); model(b); 
            if device.type=="cuda": torch.cuda.synchronize()
            latencies.append((time.perf_counter()-t)*1000)
    avg_aspects=len(eval_examples)/max(1,len({x.example_id for x in eval_examples}))
    pair_lengths=[len(tokenizer.encode(ex.text,ex.aspect,truncation=True,max_length=256)) for ex in eval_examples]
    avg_pair_tokens=float(np.mean(pair_lengths))
    pair_flops=analytic_flops(round(avg_pair_tokens),model.encoder.config.hidden_size,12,12*avg_aspects,
                              avg_aspects,1.)
    row={"method":"B1_pair_input","seed":args.seed,"tau":"","alpha":"","beta":"","split":split_note,
         "RMSE_V":metrics["RMSE_V"],"RMSE_A":metrics["RMSE_A"],"RMSE_VA":metrics["RMSE_VA"],
         "Pearson_V":metrics["Pearson_V"],"Pearson_A":metrics["Pearson_A"],
         "avg_aspect_layers":12.,"avg_sentence_layers":12.*avg_aspects,"FLOPs":pair_flops,
         "latency_ms_median":float(np.median(latencies)),"latency_ms_p95":float(np.percentile(latencies,95)),
         "device":str(device)}
    cdir=root/("results_smoke/checkpoints" if args.quick else "checkpoints"); cdir.mkdir(parents=True,exist_ok=True); name=f"B1_pair_input_seed{args.seed}"
    torch.save({"state_dict":model.state_dict(),"encoder":args.encoder,"seed":args.seed},cdir/f"{name}.pt")
    (cdir/f"{name}.json").write_text(json.dumps({"config":vars(args),"split":split_note,"metrics":row},indent=2),encoding="utf-8")
    out=root/("results_smoke" if args.quick else "results")/"runs.csv"; out.parent.mkdir(exist_ok=True); exists=out.exists()
    import csv
    with out.open("a",newline="",encoding="utf-8") as f:
        if exists:
            with out.open(newline="",encoding="utf-8") as existing_file: fields=next(csv.reader(existing_file))
        else: fields=list(row)
        w=csv.DictWriter(f,fieldnames=fields)
        if not exists: w.writeheader()
        w.writerow(row)
    print(f"saved {cdir/f'{name}.pt'} and {out}",flush=True)

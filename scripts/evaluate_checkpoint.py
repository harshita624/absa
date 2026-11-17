"""Run threshold, fixed-layer, variance, latency and RQ3 evaluation for a checkpoint."""
from __future__ import annotations
import argparse,csv,sys,time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from transformers import AutoTokenizer
from absa_depth.data import load_records,AspectDataset,SentenceCollator
from absa_depth.model import AdaptiveDepthModel
from absa_depth.train import make_loader,select_device,_to_device
from absa_depth.metrics import regression_metrics,unnormalize
from absa_depth.infer import simulate_exit,simulate_variance_exit,predict_true_early
from absa_depth.cost import analytic_flops
from absa_depth.analysis import subgroup_masks,subgroup_summary


def collect(model,loader,device):
    """Collect full-depth outputs once so all offline exit rules share predictions."""
    ys=[];mus=[];lvs=[];gates=[];groups=[];texts=[];ids=[];aspects=[];offset=0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch=_to_device(batch,device); mu,lv,lam,_=model(batch)
            ys.append(batch["targets"].cpu().numpy())
            mus.append(unnormalize(mu).cpu().numpy()); lvs.append(lv.cpu().numpy()); gates.append(lam.cpu().numpy())
            groups.append(batch["sentence_index"].cpu().numpy()+offset)
            texts.extend(batch["texts"]); ids.extend(batch["sentence_ids"]); aspects.extend(batch["aspects"])
            offset+=len(batch["texts"])
    return np.concatenate(ys),np.concatenate(mus,axis=1),np.concatenate(lvs,axis=1),np.concatenate(gates,axis=1),np.concatenate(groups),texts,ids,aspects


def pack_depth(depths,groups,sentence_count):
    sent=np.asarray([int(np.max(depths[groups==i])) for i in range(sentence_count)])
    return float(np.mean(depths)),float(np.mean(sent)),sent


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--checkpoint",required=True);p.add_argument("--method",default="ours")
    p.add_argument("--seed",type=int,default=42);p.add_argument("--encoder",default="bert-base-uncased")
    p.add_argument("--latency",action="store_true");p.add_argument("--tau",type=float,default=.5)
    p.add_argument("--no_gate_variance",action="store_true")
    a=p.parse_args();root=Path(a.root).resolve();device=select_device()
    files=sorted(x for x in (root/"data").rglob("*") if x.suffix.lower() in {".json",".jsonl"}
                 and "synthetic" not in x.parts
                 and ("dev" in x.stem.lower() or "valid" in x.stem.lower()))
    if not files: raise FileNotFoundError("No official dev/validation JSON/JSONL files found under data/.")
    examples=[e for f in files for e in load_records(f)]
    tokenizer=AutoTokenizer.from_pretrained(a.encoder,use_fast=True)
    model=AdaptiveDepthModel(a.encoder,gate_use_variance=not a.no_gate_variance).to(device)
    state=torch.load(root/a.checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(state["state_dict"]);model.eval()
    loader=make_loader(examples,tokenizer,32,False)
    y,mu,lv,lam,group,texts,ids,aspects=collect(model,loader,device)
    n_sent=len(texts); counts=np.bincount(group,minlength=n_sent); n_layers=mu.shape[0]
    avg_tokens=float(np.mean([len(tokenizer.encode(t,truncation=True,max_length=256)) for t in texts]))
    rows=[]
    def add_row(policy,label,pred,depth):
        metrics=regression_metrics(pred,y); aa,ss,_=pack_depth(depth,group,n_sent)
        flops=analytic_flops(round(avg_tokens),model.hidden_size,n_layers,ss,float(np.mean(counts)),aa)
        rows.append({"method":a.method,"seed":a.seed,"policy":policy,"setting":label,**metrics,
                     "avg_aspect_layers":aa,"avg_sentence_layers":ss,"FLOPs_per_sentence_est":flops,
                     "latency_ms_median":"","latency_ms_p95":"",
                     "device":str(device),"n_aspects":len(y),"n_sentences":n_sent})
    last=mu[-1]
    add_row("last_layer","12",last,np.full(len(y),n_layers,dtype=int))
    tau_values=[.1,.2,.3,.4,.5,.6,.7,.8,.9,.95]
    tau_outputs={}
    for tau in tau_values:
        pp,dd=simulate_exit(torch.tensor(mu),torch.tensor(lam),tau)
        pred=pp.numpy();depth=dd.numpy();tau_outputs[tau]=(pred,depth)
        add_row("learned_gate_tau",str(tau),pred,depth)
    for k in [2,4,6,8,10,12]:
        add_row("fixed_layer",str(k),mu[k-1],np.full(len(y),k,dtype=int))
    theta_grid=[.05,.1,.25,.5,1.,2.,4.,8.,16.]
    for theta in theta_grid:
        pp,dd=simulate_variance_exit(torch.tensor(mu),torch.tensor(lv),theta)
        add_row("variance_threshold",str(theta),pp.numpy(),dd.numpy())
    results=root/"results";results.mkdir(exist_ok=True)
    out=results/f"sweeps_{a.method}_seed{a.seed}.csv"
    # tau=1.0 is used to measure the full-depth reference at batch size one;
    # no sigmoid gate can exceed 1, so its selected output is the last layer.
    selected_pred,selected_depth=tau_outputs.get(
        a.tau,(last,np.full(len(y),n_layers,dtype=int)))
    masks=subgroup_masks(texts,counts.tolist(),y)
    rq=[]
    for name,mask in masks.items():
        rq.append({"subgroup":name,**subgroup_summary(mask,y,last,selected_pred,selected_depth)})
    with (results/f"rq3_{a.method}_seed{a.seed}.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rq[0]));w.writeheader();w.writerows(rq)
    if a.latency:
        groups_by_id=OrderedDict()
        for ex in examples: groups_by_id.setdefault(ex.example_id,{"text":ex.text,"aspects":[]})["aspects"].append(ex.aspect)
        items=list(groups_by_id.values())
        for item in items[:10]:
            predict_true_early(model,tokenizer,item["text"],item["aspects"],a.tau,device)
        times=[]
        for item in items:
            if device.type=="cuda":torch.cuda.synchronize()
            t=time.perf_counter()
            predict_true_early(model,tokenizer,item["text"],item["aspects"],a.tau,device)
            if device.type=="cuda":torch.cuda.synchronize()
            times.append((time.perf_counter()-t)*1000)
        latency={"method":a.method,"seed":a.seed,"tau":a.tau,"device":str(device),
                 "median_ms":float(np.median(times)),"p95_ms":float(np.percentile(times,95)),
                 "n_sentences":len(times),"warmup_sentences":min(10,len(items))}
        with (results/f"latency_{a.method}_seed{a.seed}.json").open("w",encoding="utf-8") as f:
            import json;json.dump(latency,f,indent=2)
        for row in rows:
            if row["policy"]=="learned_gate_tau" and float(row["setting"])==a.tau:
                row["latency_ms_median"]=latency["median_ms"]
                row["latency_ms_p95"]=latency["p95_ms"]
            elif a.tau>=1.0 and row["policy"]=="last_layer":
                row["latency_ms_median"]=latency["median_ms"]
                row["latency_ms_p95"]=latency["p95_ms"]
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Refresh the run summary row with official-dev scores, analytic cost and measured latency.
    run_csv=results/"runs.csv"
    if run_csv.exists():
        with run_csv.open(newline="",encoding="utf-8") as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;run_rows=list(reader)
        chosen=next((r for r in rows if r["policy"]=="learned_gate_tau" and float(r["setting"])==a.tau),None)
        if chosen is None and a.tau>=1.0: chosen=next((r for r in rows if r["policy"]=="last_layer"),None)
        if chosen:
            for r in run_rows:
                if r.get("method")==a.method and r.get("seed")==str(a.seed):
                    for key in ("RMSE_V","RMSE_A","RMSE_VA","Pearson_V","Pearson_A",
                                "avg_aspect_layers","avg_sentence_layers","FLOPs",
                                "latency_ms_median","latency_ms_p95"):
                        if key in fields:
                            source="FLOPs_per_sentence_est" if key=="FLOPs" else key
                            if source in chosen and chosen[source]!="":r[key]=chosen[source]
            with run_csv.open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(run_rows)
    print(f"Wrote {out}, RQ3 subgroup table, and {'latency results' if a.latency else 'no latency (pass --latency)'}")

if __name__=="__main__":main()

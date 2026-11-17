"""Aggregate per-run CSVs into mean and standard deviation by method."""
from __future__ import annotations
import argparse, csv, statistics
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default=str(Path(__file__).resolve().parents[1])); a=p.parse_args()
    root=Path(a.root); run_file=root/"results"/"runs.csv"
    rows=list(csv.DictReader(run_file.open(encoding="utf-8"))) if run_file.exists() else []
    if not rows: print("No runs found; no aggregate written."); return
    metrics=["RMSE_V","RMSE_A","RMSE_VA","avg_aspect_layers","avg_sentence_layers","FLOPs","latency_ms_median"]
    groups={}
    for row in rows: groups.setdefault(row["method"],[]).append(row)
    out=root/"results"/"aggregate.csv"; out.parent.mkdir(exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["method","n"]+[x+s for x in metrics for s in ("_mean","_std")]); writer.writeheader()
        for method,items in groups.items():
            result={"method":method,"n":len(items)}
            for m in metrics:
                vals=[]
                for r in items:
                    try: vals.append(float(r[m]))
                    except (ValueError,TypeError,KeyError): pass
                result[m+"_mean"]=statistics.mean(vals) if vals else ""
                result[m+"_std"]=statistics.stdev(vals) if len(vals)>1 else (0 if vals else "")
            writer.writerow(result)
    print(f"Wrote {out}")
if __name__=="__main__": main()

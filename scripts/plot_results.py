"""Plot executed experiment sweeps, RQ3 depth bars and beta ablation results."""
from __future__ import annotations
import argparse,csv,statistics
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

def read_csv(path):
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def num(row,key):
    try:return float(row[key])
    except (ValueError,TypeError,KeyError):return None
def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default=str(Path(__file__).resolve().parents[1]));a=p.parse_args()
    root=Path(a.root);out=root/"figures";out.mkdir(exist_ok=True)
    sweeps=[r for f in (root/"results").glob("sweeps_*.csv") for r in read_csv(f)]
    if sweeps:
        fig,ax=plt.subplots()
        for policy in ("learned_gate_tau","variance_threshold","fixed_layer"):
            groups=defaultdict(list)
            for row in sweeps:
                # Compare OURS' learned gate with B4 thresholds and B3 fixed
                # layers from the deep-supervised model only.
                expected_method = ("ours" if policy=="learned_gate_tau" else "deep_supervised")
                if row["policy"]==policy and row.get("method")==expected_method:
                    x=num(row,"avg_aspect_layers");y=num(row,"RMSE_VA")
                    if x is not None and y is not None:groups[x].append(y)
            if groups:
                pts=sorted((x,statistics.mean(ys)) for x,ys in groups.items())
                ax.plot([x for x,_ in pts],[y for _,y in pts],"o-",label=policy.replace("_"," "))
        ax.set(xlabel="Average layers per aspect",ylabel="RMSE_VA (1–9 scale)",title="Error versus compute depth")
        ax.legend();fig.tight_layout();fig.savefig(out/"rmse_vs_depth.png",dpi=160);plt.close(fig)
    subgroup_files=list((root/"results").glob("rq3_ours_*.csv"))
    if subgroup_files:
        groups=defaultdict(list)
        for f in subgroup_files:
            for row in read_csv(f):
                v=num(row,"mean_exit_layer")
                if v is not None:groups[row["subgroup"]].append(v)
        labels=list(groups);vals=[statistics.mean(groups[x]) for x in labels]
        fig,ax=plt.subplots();ax.bar(labels,vals);ax.set(ylabel="Mean exit layer",title="Exit depth by aspect subgroup");ax.tick_params(axis="x",rotation=20);fig.tight_layout();fig.savefig(out/"subgroup_exit_depth.png",dpi=160);plt.close(fig)
    run_file=root/"results"/"runs.csv"
    if run_file.exists():
        rows=read_csv(run_file);beta=defaultdict(list)
        for r in rows:
            if r.get("method","").startswith("beta_"):
                x=num(r,"beta");y=num(r,"RMSE_VA")
                if x is not None and y is not None:beta[x].append(y)
        if beta:
            pts=sorted((x,statistics.mean(ys)) for x,ys in beta.items())
            fig,ax=plt.subplots();ax.plot([x for x,_ in pts],[y for _,y in pts],"o-")
            ax.set(xlabel="Depth penalty beta",ylabel="RMSE_VA (1–9 scale)",title="Beta sweep");fig.tight_layout();fig.savefig(out/"beta_sweep.png",dpi=160);plt.close(fig)
    if not sweeps and not subgroup_files and not run_file.exists():print("No executed results found; no figures generated.")
    else:print(f"Figures written under {out}")
if __name__=="__main__":main()

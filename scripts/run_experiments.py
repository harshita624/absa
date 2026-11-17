"""Cross-platform experiment launcher for pilot, core and full experiment suites."""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

def invoke(root, args, method, seed, *, two_stage=False, no_variance=False, alpha=None, beta=None):
    cmd=[sys.executable,"-m","absa_depth.train","--root",str(root),"--seed",str(seed),
         "--epochs",str(args.epochs),"--batch_size",str(args.batch_size),"--method",method]
    if args.max_train_sentences: cmd += ["--max_train_sentences",str(args.max_train_sentences)]
    if two_stage: cmd.append("--two_stage")
    if no_variance: cmd.append("--no_gate_variance")
    if alpha is not None: cmd += ["--alpha",str(alpha)]
    if beta is not None: cmd += ["--beta",str(beta)]
    subprocess.run(cmd,check=True,env={**os.environ,"PYTHONPATH":str(root/"src")})

def main():
    p=argparse.ArgumentParser(); p.add_argument("--quick",action="store_true")
    p.add_argument("--suite",choices=["core","full"],default="core")
    p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    p.add_argument("--epochs",type=int,default=3); p.add_argument("--batch_size",type=int,default=32)
    p.add_argument("--max_train_sentences",type=int,default=0)
    p.add_argument("--root",default=str(Path(__file__).resolve().parents[1])); a=p.parse_args(); root=Path(a.root).resolve()
    if a.quick:
        subprocess.run([sys.executable,str(root/"scripts"/"make_synthetic_data.py")],check=True)
        subprocess.run([sys.executable,"-m","absa_depth.train","--root",str(root),"--quick"],check=True,
                       env={**os.environ,"PYTHONPATH":str(root/"src")})
        return
    for seed in a.seeds:
        if a.suite=="core":
            invoke(root,a,"ours",seed,two_stage=True)
            invoke(root,a,"deep_supervised",seed)
            invoke(root,a,"B1_pair_input",seed)
            invoke(root,a,"B2_last",seed)
        else:
            invoke(root,a,"ours",seed,two_stage=True)
            invoke(root,a,"deep_supervised",seed)
            invoke(root,a,"B1_pair_input",seed)
            invoke(root,a,"B2_last",seed)
            invoke(root,a,"ours_no_variance",seed,no_variance=True)
            invoke(root,a,"ours_no_deep_supervision",seed,alpha=0.0)
            for beta in (0.0,0.001,0.003,0.01,0.03):
                invoke(root,a,f"beta_{beta:g}",seed,beta=beta)
        eval_methods=("ours","deep_supervised","B2_last")
        if a.suite=="full": eval_methods += ("ours_no_variance","ours_no_deep_supervision")
        for method in eval_methods:
            checkpoint=f"checkpoints/{method}_seed{seed}.pt"
            command=[sys.executable,str(root/"scripts"/"evaluate_checkpoint.py"),"--root",str(root),
                     "--checkpoint",checkpoint,"--method",method,"--seed",str(seed)]
            if method=="ours": command += ["--latency","--tau","0.5"]
            elif method in {"deep_supervised","B2_last"}: command += ["--latency","--tau","1.0"]
            if method=="ours_no_variance": command.append("--no_gate_variance")
            subprocess.run(command,check=True,env={**os.environ,"PYTHONPATH":str(root/"src")})
    subprocess.run([sys.executable,str(root/"scripts"/"aggregate_results.py"),"--root",str(root)],check=True)
    subprocess.run([sys.executable,str(root/"scripts"/"plot_results.py"),"--root",str(root)],check=True)
if __name__=="__main__": main()

"""Run an isolated, reduced-data three-seed ABSA pilot for CPU environments.

This is a fast pilot, not a substitute for the complete full-data suite. It uses
up to 512 training sentences per seed, two epochs for OURS, and one epoch for
each core baseline. All outputs are isolated under fast_pilot/.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    run_root = project / "fast_pilot"
    data_root = run_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    official_files = sorted(
        p for p in (project / "data").rglob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
        and "synthetic" not in p.parts
    )
    if not official_files:
        raise FileNotFoundError("No real JSON/JSONL data found under data/.")
    for source in official_files:
        target = data_root / source.name
        if not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
            shutil.copy2(source, target)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HUB_OFFLINE"] = "1"
    env["USE_TF"] = "0"

    def run(command: list[str]) -> None:
        print("\n$ " + " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=project, env=env)

    for seed in (42, 43, 44):
        common = ["--root", str(run_root), "--seed", str(seed), "--batch_size", "32",
                  "--max_train_sentences", "512"]
        training = (("ours", 2, True), ("deep_supervised", 1, False),
                    ("B2_last", 1, False), ("B1_pair_input", 1, False))
        for method, epochs, two_stage in training:
            checkpoint=run_root/"checkpoints"/f"{method}_seed{seed}.pt"
            if checkpoint.exists():
                print(f"\nSkipping completed training checkpoint: {checkpoint}",flush=True)
                continue
            command=[sys.executable,"-m","absa_depth.train",*common,
                     "--epochs",str(epochs),"--method",method]
            if two_stage: command.append("--two_stage")
            run(command)

        for method, tau, latency in (("ours", "0.5", True),
                                     ("deep_supervised", "1.0", True),
                                     ("B2_last", "1.0", True)):
            command = [sys.executable, str(project / "scripts" / "evaluate_checkpoint.py"),
                       "--root", str(run_root), "--checkpoint", f"checkpoints/{method}_seed{seed}.pt",
                       "--method", method, "--seed", str(seed), "--tau", tau]
            if latency:
                command.append("--latency")
            sweep=run_root/"results"/f"sweeps_{method}_seed{seed}.csv"
            latency_file=run_root/"results"/f"latency_{method}_seed{seed}.json"
            if sweep.exists() and (not latency or latency_file.exists()):
                print(f"\nSkipping completed evaluation: {sweep}",flush=True)
            else:
                run(command)

    run([sys.executable, str(project / "scripts" / "aggregate_results.py"), "--root", str(run_root)])
    run([sys.executable, str(project / "scripts" / "plot_results.py"), "--root", str(run_root)])
    print(f"\nFast pilot complete. Outputs are isolated in: {run_root}", flush=True)
    print("This reduced-data pilot must not be reported as the full-data experiment.", flush=True)


if __name__ == "__main__":
    main()

"""Create a deterministic tiny dataset matching the expected annotation schema."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",default=None); p.add_argument("--count",type=int,default=60); a=p.parse_args()
    root=Path(__file__).resolve().parents[1]; out=Path(a.out) if a.out else root/"data"/"synthetic"
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i in range(a.count):
        text=f"The food was {'excellent' if i%2 else 'bland'} but the service was {'slow' if i%3 else 'attentive'}."
        rows.append({"ID":f"syn-{i:04d}","Text":text,"Aspects":[
            {"Aspect":"food","VA":f"{7.0 if i%2 else 3.5:.2f}#6.00"},
            {"Aspect":"service","VA":f"{3.0 if i%3 else 7.0:.2f}#5.50"}]})
    split=max(2,int(.8*len(rows)))
    for name,chunk in (("train",rows[:split]),("dev",rows[split:])):
        (out/f"{name}.json").write_text(json.dumps(chunk,indent=2),encoding="utf-8")
    print(f"Wrote {len(rows)} synthetic sentences to {out} (train={split}, dev={len(rows)-split}). Use --quick to save runs to results_smoke/.")
if __name__=="__main__": main()

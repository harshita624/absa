"""Discover JSON/JSONL files and print their schema and aspect counts."""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from absa_depth.data import discover_data, load_records

def main():
    root=Path(__file__).resolve().parents[1]/"data"; files=discover_data(root)
    if not files:
        print(f"No JSON/JSONL data found under {root}. Add official train/dev data or run make_synthetic_data.py.")
        return
    for path in files:
        raw=path.read_text(encoding="utf-8-sig").splitlines()
        sample=next((x for x in raw if x.strip()),"{}")
        try: obj=json.loads(sample if path.suffix==".jsonl" else path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as e: print(f"{path}: invalid JSON: {e}"); continue
        examples=load_records(path); by_id=Counter(x.example_id for x in examples)
        print(f"\n{path.relative_to(root)}\n top-level type: {type(obj).__name__}; parsed aspects: {len(examples)}")
        print(f" sentences: {len(by_id)}; aspects/sentence: min={min(by_id.values(),default=0)} max={max(by_id.values(),default=0)} mean={sum(by_id.values())/max(1,len(by_id)):.2f}")
        print(" schema sample:",json.dumps(obj[0] if isinstance(obj,list) and obj else obj,ensure_ascii=False)[:2000])
        for ex in examples[:3]: print(" example:",ex)
if __name__=="__main__": main()

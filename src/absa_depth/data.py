"""Data discovery, tolerant DimABSA parsing, tokenization and batching."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Inspection and schema diagnostics do not require ML dependencies.
    torch = None
    class Dataset:
        def __class_getitem__(cls, item): return cls


@dataclass(frozen=True)
class AspectExample:
    text: str
    aspect: str
    valence: float
    arousal: float
    example_id: str = ""


def _pick(mapping: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    lower = {str(k).lower().replace("_", ""): v for k, v in mapping.items()}
    for name in names:
        key = name.lower().replace("_", "")
        if key in lower:
            return lower[key]
    return default


def _parse_va(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try: return float(value[0]), float(value[1])
        except (TypeError, ValueError): return None
    if isinstance(value, dict):
        v = _pick(value, ("valence", "v")); a = _pick(value, ("arousal", "a"))
        try: return float(v), float(a)
        except (TypeError, ValueError): return None
    if isinstance(value, str):
        parts = value.replace(",", "#").split("#")
        if len(parts) >= 2:
            try: return float(parts[0].strip()), float(parts[1].strip())
            except ValueError: return None
    return None


def records_from_object(obj: Any) -> list[dict[str, Any]]:
    """Normalize a JSON object/list into sentence records, tolerating wrapper keys."""
    if isinstance(obj, list): return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("data", "records", "sentences", "examples", "instances"):
            value = _pick(obj, (key,))
            if isinstance(value, list): return [x for x in value if isinstance(x, dict)]
        return [obj]
    return []


def load_records(path: str | Path) -> list[AspectExample]:
    """Load JSON/JSONL records and flatten aspect annotations."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        objects = [json.loads(line) for line in raw.splitlines() if line.strip()]
        records = [rec for obj in objects for rec in records_from_object(obj)]
    else: records = records_from_object(json.loads(raw))
    examples: list[AspectExample] = []
    for i, rec in enumerate(records):
        text = _pick(rec, ("text", "sentence", "review", "content"))
        if not isinstance(text, str): continue
        rid = str(_pick(rec, ("id", "uid", "sentence_id"), i))
        # Official DimABSA dev rows use Aspect_VA; train_alltasks rows may
        # carry the same Aspect/VA pair inside Triplet or Quadruplet objects.
        aspects = _pick(rec, ("aspect_va", "aspects", "aspect_terms", "targets", "triplet", "quadruplet"), [])
        if isinstance(aspects, dict): aspects = [aspects]
        if not isinstance(aspects, list): continue
        for aspect in aspects:
            if isinstance(aspect, str):
                term, va = aspect, _pick(rec, ("va", "label", "score"))
            elif isinstance(aspect, dict):
                term = _pick(aspect, ("aspect", "term", "target", "aspect_term"), "NULL")
                va = _pick(aspect, ("va", "label", "score", "sentiment", "rating"))
                if va is None:
                    va = (_pick(aspect, ("valence", "v")), _pick(aspect, ("arousal", "a")))
            else: continue
            parsed = _parse_va(va)
            if parsed is None or not isinstance(term, str): continue
            examples.append(AspectExample(text, term, *parsed, rid))
    return examples


class AspectDataset(Dataset[AspectExample]):
    """Flat aspect dataset; sentence IDs preserve grouping for evaluation."""
    def __init__(self, examples: list[AspectExample]): self.examples = examples
    def __len__(self) -> int: return len(self.examples)
    def __getitem__(self, index: int) -> AspectExample: return self.examples[index]


class SentenceCollator:
    """Batch sentence groups, encode each distinct sentence once, flatten aspects."""
    def __init__(self, tokenizer: Any, max_length: int = 256):
        self.tokenizer, self.max_length = tokenizer, max_length
    def __call__(self, examples: list[AspectExample]) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("SentenceCollator requires PyTorch; install requirements.txt first.")
        grouped: dict[str, list[AspectExample]] = {}
        for ex in examples: grouped.setdefault(ex.example_id or f"anon-{len(grouped)}", []).append(ex)
        sentences: list[str] = []; ids: list[str] = []; flat: list[tuple[int, AspectExample]] = []
        lookup: dict[tuple[str, str], int] = {}
        for key, group in grouped.items():
            text = group[0].text; compound = (key, text)
            si = lookup.setdefault(compound, len(sentences))
            if si == len(sentences): sentences.append(text); ids.append(key)
            flat.extend((si, ex) for ex in group)
        encoded = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_length,
                                 return_offsets_mapping=True, return_tensors="pt")
        offsets = encoded.pop("offset_mapping")
        starts, ends, sentence_index = [], [], []
        for si, ex in flat:
            start = end = 0
            if ex.aspect and ex.aspect.upper() != "NULL":
                pos = sentences[si].casefold().find(ex.aspect.casefold())
                if pos >= 0: start, end = pos, pos + len(ex.aspect)
            starts.append(start); ends.append(end); sentence_index.append(si)
        return {**encoded, "offset_mapping": offsets, "texts": sentences, "sentence_ids": ids,
                "aspects": [ex.aspect for _, ex in flat],
                "targets": torch.tensor([[ex.valence, ex.arousal] for _, ex in flat], dtype=torch.float32),
                "sentence_index": torch.tensor(sentence_index, dtype=torch.long),
                "char_starts": torch.tensor(starts), "char_ends": torch.tensor(ends)}


def discover_data(root: str | Path) -> list[Path]:
    """Find JSON/JSONL datasets beneath root, excluding generated smoke outputs."""
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
                  and "results_smoke" not in p.parts)

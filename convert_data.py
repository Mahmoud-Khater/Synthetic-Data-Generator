import os
import json
from pathlib import Path
from typing import Optional, Union

from datasets import load_dataset, Dataset


def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parquet_hf_to_jsonl(
    dataset_name: str,
    split: str = "train",
    output_path: str = "./data/real_reviews.jsonl",
    dataset_config: Optional[str] = None,
    hf_token: Optional[str] = None,
    streaming: bool = False,
    max_rows: Optional[int] = None,
    flatten_nested: bool = False,
) -> str:
    """
    Load a parquet-backed Hugging Face dataset and export it to JSONL locally.

    Args:
        dataset_name: e.g. "allenai/c4" or "username/my_dataset"
        split: e.g. "train", "validation", "test"
        output_path: where to write JSONL
        dataset_config: optional config/subset name (some datasets require it)
        hf_token: token for private datasets (or set HF_TOKEN env var)
        streaming: if True, uses streaming mode (memory-friendly)
        max_rows: optionally limit number of rows exported
        flatten_nested: if True, tries to flatten nested structures to be JSON-serializable

    Returns:
        Absolute path to the written JSONL file.
    """
    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    # Load dataset (parquet is often the storage format behind the scenes)
    ds = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        token=token,
        streaming=streaming,
    )

    out_file = Path(output_path)
    _ensure_dir(out_file.parent)

    def make_jsonable(x):
        """
        Convert non-JSON-serializable objects (like numpy types) to plain Python types.
        """
        # datasets can include numpy scalars, bytes, etc.
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if isinstance(x, bytes):
            # best-effort: decode bytes as utf-8 if possible
            try:
                return x.decode("utf-8")
            except Exception:
                return list(x)
        if isinstance(x, dict):
            return {k: make_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [make_jsonable(v) for v in x]
        # numpy scalars, pandas types, etc.
        try:
            import numpy as np  # optional

            if isinstance(x, np.generic):
                return x.item()
        except Exception:
            pass
        # fallback: string representation
        return str(x)

    def maybe_flatten(example: dict) -> dict:
        if not flatten_nested:
            return example
        # simple flatten: only flattens one level of dict nesting
        flat = {}
        for k, v in example.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}.{kk}"] = vv
            else:
                flat[k] = v
        return flat

    written = 0
    with out_file.open("w", encoding="utf-8") as f:
        # Streaming datasets return IterableDataset; normal returns Dataset
        for ex in ds:
            ex = maybe_flatten(ex)
            ex = make_jsonable(ex)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            written += 1
            if max_rows is not None and written >= max_rows:
                break

    return str(out_file.resolve())


if __name__ == "__main__":

    path = parquet_hf_to_jsonl(
        dataset_name="juliensimon/amazon-shoe-reviews",
        split="train",                
        output_path="./out/shoes.jsonl",
        streaming=True,                
        max_rows=None,                 
        flatten_nested=False
    )

    print(f"Saved JSONL to: {path}")


from pathlib import Path
import pandas as pd
import re
from typing import List, Dict, Any, Optional
from datavis.io_utils import read_jsonl

def extract_pred_and_prompt(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    def pick_pred(d: Dict[str, Any]) -> str:
        for k in ["prediction", "generated", "code", "output", "generation"]:
            if k in d and isinstance(d[k], str):
                return d[k]
        for k in ["prediction", "generated", "output", "generation"]:
            if k in d and isinstance(d[k], dict) and isinstance(d[k].get("text"), str):
                return d[k]["text"]
        return ""

    def pick_prompt_type(d: Dict[str, Any]) -> Optional[str]:
        for k in ["prompt_type", "promptVersion", "prompt", "template_id"]:
            if k in d and isinstance(d[k], str):
                return d[k]
        return None

    def pick_id(d: Dict[str, Any]) -> Optional[str]:
        for k in ["id", "qid", "sample_id", "problem_id", "uid"]:
            if k in d:
                return str(d[k])
        return None

    recs = []
    for d in rows:
        recs.append({
            "join_id": pick_id(d),
            "prediction": pick_pred(d),
            "prompt_type": pick_prompt_type(d)
        })
    df = pd.DataFrame(recs)

    if df["join_id"].isna().any():
        df["__row_index__"] = range(len(df))
        df["join_id"] = df["join_id"].fillna(df["__row_index__"].astype(str))
        df.drop(columns=["__row_index__"], inplace=True)
    return df

def load_generation_file(file_path: Path, variant_hint: str) -> pd.DataFrame:
    rows = read_jsonl(file_path)
    dfp = extract_pred_and_prompt(rows)
    dfp["variant"] = variant_hint
    return dfp

def load_generation_dir(dir_path: Path, variant_hint: str) -> pd.DataFrame:
    all_parts = []
    for fp in sorted(dir_path.glob("*.jsonl")):
        rows = read_jsonl(fp)
        dfp = extract_pred_and_prompt(rows)
        if dfp["prompt_type"].isna().all():
            m = re.search(r"(v\d+(_\d+)?)", fp.stem.lower())
            dfp["prompt_type"] = m.group(1) if m else "unknown"
        dfp["variant"] = variant_hint
        dfp["__source_file__"] = str(fp)
        all_parts.append(dfp)
    return pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame(columns=["join_id","prediction","prompt_type","variant"])

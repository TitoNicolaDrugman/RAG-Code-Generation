from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                try:
                    rows.append(json.loads(line.encode("utf-8", "ignore").decode("utf-8")))
                except Exception:
                    pass
    return rows

def load_dataset(df_path: Path) -> pd.DataFrame:
    data = read_jsonl(df_path)
    df = pd.DataFrame(data)

    candidate_ref_cols = ["references", "reference", "ground_truth", "solution", "target", "code"]
    ref_col = None
    for c in candidate_ref_cols:
        if c in df.columns:
            ref_col = c
            break
    if ref_col is None:
        raise ValueError(f"Non trovo colonna reference in {df_path}")

    def to_list_refs(x):
        if isinstance(x, list):
            return [s for s in x if isinstance(s, str)]
        if isinstance(x, str):
            return [x]
        return []

    df["references_norm"] = df[ref_col].apply(to_list_refs)

    id_col = None
    for c in ["id", "qid", "sample_id", "problem_id", "uid"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        df["__row_index__"] = np.arange(len(df))
        id_col = "__row_index__"

    df = df[[id_col, "references_norm"]].rename(columns={id_col: "join_id"})
    return df

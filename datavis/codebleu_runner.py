from pathlib import Path
import pandas as pd
from datavis.io_utils import load_dataset
from datavis.generation_loader import load_generation_file, load_generation_dir
from codebleu_F import compute_grouped_codebleu

def run_codebleu(
    dataset_file: Path,
    baseline_file: Path,
    bm25_file: Path,
    cosine_file: Path,
    hybrid_file: Path,
    multihop_decomp_dir: Path,
    multihop_iter_dir: Path,
    out_csv: Path,
    average: str = "macro"
) -> pd.DataFrame:
    dataset_df = load_dataset(dataset_file)

    frames = []
    frames.append(load_generation_file(baseline_file, variant_hint="baseline"))
    frames.append(load_generation_file(bm25_file,   variant_hint="bm25"))
    frames.append(load_generation_file(cosine_file, variant_hint="cosine"))
    frames.append(load_generation_file(hybrid_file, variant_hint="hybrid"))
    frames.append(load_generation_dir(multihop_decomp_dir, variant_hint="multihop_decomposition"))
    frames.append(load_generation_dir(multihop_iter_dir,   variant_hint="multihop_iterative"))

    gens_df = pd.concat(frames, ignore_index=True)

    if "prompt_type" not in gens_df.columns:
        gens_df["prompt_type"] = "unknown"
    else:
        gens_df["prompt_type"] = gens_df["prompt_type"].fillna("unknown")

    dataset_df["join_id"] = dataset_df["join_id"].astype(str)
    gens_df["join_id"]    = gens_df["join_id"].astype(str)
    merged = gens_df.merge(dataset_df, on="join_id", how="left")

    if merged["references_norm"].isna().any():
        mask = merged["references_norm"].isna()
        fallback_refs = dataset_df["references_norm"].tolist()
        merged.loc[mask, "references_norm"] = merged.loc[mask].index.map(
            lambda i: fallback_refs[i % len(fallback_refs)]
        )

    summary = compute_grouped_codebleu(
        merged,
        prediction_col="prediction",
        references_col="references_norm",
        group_cols=("variant","prompt_type"),
        average=average,
    )

    summary = summary.sort_values(["variant","prompt_type"]).reset_index(drop=True)
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Salvato:", out_csv)
    return summary

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


def load_and_prepare_df(
    file_path: str,
    cve_col: str = "CVE-ID",
    score_col: str = "Final_Score",
    label_col: str = "Label",
    sheet_name: Any = 0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Load and clean the input file, then sort with a random tie-break that
    matches the baseline script's df.sample + mergesort behaviour.
    """
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel.")

    required_cols = {cve_col, score_col, label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    df = df.dropna(subset=[cve_col, score_col]).copy()
    if df.empty:
        raise ValueError("Input data is empty after cleaning.")

    # Random tie-break: shuffle first, then stable-sort by score, so equal-score
    # rows end up in the random order produced by `random_seed`.
    df = df.sample(frac=1, random_state=random_seed)
    df = df.sort_values(
        by=[cve_col, score_col],
        ascending=[True, False],
        kind='mergesort',
    ).copy()

    df["Rank"] = df.groupby(cve_col).cumcount() + 1

    return df


def build_per_cve_records(
    df: pd.DataFrame,
    cve_col: str = "CVE-ID",
    label_col: str = "Label",
    ks: List[int] = [1, 3, 5, 10],
) -> List[Dict[str, Any]]:
    """
    Build a per-CVE ranking record. Each entry contains:
      - total_positive: number of positives for that CVE
      - tp_at_k:        true positives within top-K, for each K in `ks`
      - rr:             reciprocal rank of the first positive (0 if none)
    """
    records = []

    for cve_id, group in df.groupby(cve_col, sort=False):
        labels = group[label_col].to_numpy()
        total_positive = int(labels.sum())

        positive_positions = np.where(labels == 1)[0]
        if len(positive_positions) > 0:
            first_rank = int(positive_positions[0] + 1)
            rr = 1.0 / first_rank
        else:
            rr = 0.0

        tp_at_k = {k: int(labels[:k].sum()) for k in ks}

        records.append({
            "cve_id": cve_id,
            "total_positive": total_positive,
            "tp_at_k": tp_at_k,
            "rr": rr,
        })

    if not records:
        raise ValueError("No CVE records were built.")

    return records


def summarize_baseline_style_metrics(
    records: List[Dict[str, Any]],
    ks: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Aggregate per-CVE records into baseline-style ranking metrics."""
    num_cves = len(records)
    if num_cves == 0:
        raise ValueError("records is empty.")

    total_positive = sum(r["total_positive"] for r in records)
    rr_values = [r["rr"] for r in records]

    results = {
        "Num_CVEs": num_cves,
        "Total_Positive_Associations": total_positive,
        "MRR": float(np.mean(rr_values)) if rr_values else 0.0,
    }

    for k in ks:
        tp_at_k = sum(r["tp_at_k"][k] for r in records)

        recall_at_k = tp_at_k / total_positive if total_positive > 0 else 0.0
        precision_at_k = tp_at_k / (num_cves * k) if num_cves > 0 else 0.0

        if precision_at_k + recall_at_k > 0:
            f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
        else:
            f1_at_k = 0.0

        results[f"TP@{k}"] = tp_at_k
        results[f"Recall@{k}"] = recall_at_k
        results[f"Precision@{k}"] = precision_at_k
        results[f"F1@{k}"] = f1_at_k

    return results


def bootstrap_ranking_ci_baseline_style(
    records: List[Dict[str, Any]],
    ks: List[int] = [1, 3, 5, 10],
    n_bootstrap: int = 2000,
    ci: float = 95.0,
    random_seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """Bootstrap at the CVE level; metric aggregation stays baseline-style."""
    if not records:
        raise ValueError("records is empty.")

    rng = np.random.default_rng(random_seed)
    n_cves = len(records)

    point_estimates = summarize_baseline_style_metrics(records, ks)

    metrics_to_collect = ["MRR"]
    for k in ks:
        metrics_to_collect.extend([f"Precision@{k}", f"Recall@{k}", f"F1@{k}"])

    bootstrap_values = {metric: [] for metric in metrics_to_collect}

    for _ in range(n_bootstrap):
        sample_idx = rng.integers(low=0, high=n_cves, size=n_cves)
        sampled_records = [records[i] for i in sample_idx]

        sampled_metrics = summarize_baseline_style_metrics(sampled_records, ks)

        for metric in metrics_to_collect:
            bootstrap_values[metric].append(sampled_metrics[metric])

    alpha = (100.0 - ci) / 2.0
    lower_q = alpha
    upper_q = 100.0 - alpha

    ci_results = {}
    for metric, values in bootstrap_values.items():
        lower = float(np.percentile(values, lower_q))
        upper = float(np.percentile(values, upper_q))
        ci_results[metric] = (lower, upper)

    return point_estimates, ci_results


def print_results(
    point_estimates: Dict[str, float],
    ci_results: Dict[str, Tuple[float, float]],
    ks: List[int],
) -> None:
    print(f"Num_CVEs: {point_estimates['Num_CVEs']}")
    print(f"Total_Positive_Associations: {point_estimates['Total_Positive_Associations']}")
    print()

    mrr = point_estimates["MRR"]
    mrr_l, mrr_u = ci_results["MRR"]
    print(f"MRR: {mrr:.4f} (95% CI: [{mrr_l:.4f}, {mrr_u:.4f}])")
    print()

    for k in ks:
        p = point_estimates[f"Precision@{k}"]
        p_l, p_u = ci_results[f"Precision@{k}"]

        r = point_estimates[f"Recall@{k}"]
        r_l, r_u = ci_results[f"Recall@{k}"]

        f1 = point_estimates[f"F1@{k}"]
        f1_l, f1_u = ci_results[f"F1@{k}"]

        print(f"@{k}")
        print(f"  Precision@{k}: {p:.4f} (95% CI: [{p_l:.4f}, {p_u:.4f}])")
        print(f"  Recall@{k}:    {r:.4f} (95% CI: [{r_l:.4f}, {r_u:.4f}])")
        print(f"  F1@{k}:        {f1:.4f} (95% CI: [{f1_l:.4f}, {f1_u:.4f}])")
        print("-" * 30)


def run_from_file(
    file_path: str,
    cve_col: str = "CVE-ID",
    score_col: str = "Final_Score",
    label_col: str = "Label",
    sheet_name: Any = 0,
    ks: List[int] = [1, 3, 5, 10],
    n_bootstrap: int = 2000,
    random_seed: int = 42,
):
    df = load_and_prepare_df(
        file_path=file_path,
        cve_col=cve_col,
        score_col=score_col,
        label_col=label_col,
        sheet_name=sheet_name,
        random_seed=random_seed,
    )

    records = build_per_cve_records(
        df=df,
        cve_col=cve_col,
        label_col=label_col,
        ks=ks,
    )

    point_estimates, ci_results = bootstrap_ranking_ci_baseline_style(
        records=records,
        ks=ks,
        n_bootstrap=n_bootstrap,
        ci=95.0,
        random_seed=random_seed,
    )

    print_results(point_estimates, ci_results, ks)

    return df, records, point_estimates, ci_results


if __name__ == "__main__":
    file_path = "data/RankingEx/output/test_results/Ensemble_Weighted_Results_.xlsx"

    df, records, point_estimates, ci_results = run_from_file(
        file_path=file_path,
        cve_col="CVE-ID",
        score_col="Final_Score",
        label_col="Label",
        sheet_name=0,
        ks=[1, 3, 5, 10],
        n_bootstrap=2000,
        random_seed=39,
    )
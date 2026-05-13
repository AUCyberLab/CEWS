import pandas as pd
from typing import Optional, Dict, Any


def evaluate_ranking_metrics_baseline_style(
    excel_file: str,
    output_top10_file: Optional[str] = None,
    sheet_name: Any = 0,
    cve_col: str = "CVE-ID",
    score_col: str = "Final_Score",
    label_col: str = "Label",
    topk_max: int = 10,
) -> Dict[str, Any]:
    """
    Compute ranking metrics (Precision/Recall/F1 @ K and MRR) under four
    tie-breaking strategies, so the reader can see how sensitive the numbers
    are to the order of equal-scored rows.

    Strategies:
      - Original:    pandas stable sort, preserves row order in the input file
      - Optimistic:  ties broken so Label=1 comes first (best case)
      - Pessimistic: ties broken so Label=0 comes first (worst case)
      - Random:      rows shuffled before stable sort, simulating random tie-breaks
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name).copy()

    for col in (cve_col, score_col, label_col):
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")

    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=[cve_col, score_col]).copy()

    if df.empty:
        raise ValueError("Input data is empty or all key columns are null.")

    total_positive = int(df[label_col].sum())
    num_cves = df[cve_col].nunique()

    def _calculate_metrics(sorted_df: pd.DataFrame):
        sorted_df = sorted_df.copy()
        sorted_df["Rank"] = sorted_df.groupby(cve_col).cumcount() + 1

        topk_df = sorted_df[sorted_df["Rank"] <= topk_max].copy()
        metrics = {}

        for k in range(1, topk_max + 1):
            current_topk = topk_df[topk_df["Rank"] <= k]
            tp_at_k = int(current_topk[label_col].sum())

            recall_at_k = tp_at_k / total_positive if total_positive > 0 else 0.0
            precision_at_k = tp_at_k / (num_cves * k) if num_cves > 0 else 0.0

            if precision_at_k + recall_at_k > 0:
                f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
            else:
                f1_at_k = 0.0

            metrics[f"TP@{k}"] = tp_at_k
            metrics[f"Recall@{k}"] = recall_at_k
            metrics[f"Precision@{k}"] = precision_at_k
            metrics[f"F1@{k}"] = f1_at_k

        # MRR: mean of 1 / first-correct-rank, with 0 contributed by CVEs that have no positive in their list
        reciprocal_ranks = []
        for _, group in sorted_df.groupby(cve_col):
            positive_ranks = group.loc[group[label_col] == 1, "Rank"]
            if len(positive_ranks) > 0:
                reciprocal_ranks.append(1.0 / int(positive_ranks.min()))
            else:
                reciprocal_ranks.append(0.0)

        metrics["MRR"] = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        return metrics, topk_df

    # Build the four tie-breaking variants
    df_original = df.sort_values(by=[cve_col, score_col], ascending=[True, False]).copy()
    df_optimistic = df.sort_values(by=[cve_col, score_col, label_col], ascending=[True, False, False]).copy()
    df_pessimistic = df.sort_values(by=[cve_col, score_col, label_col], ascending=[True, False, True]).copy()
    # Shuffle first, then stable-sort by score, so equal-score rows end up in random order
    df_random = (
        df.sample(frac=1, random_state=39)
          .sort_values(by=[cve_col, score_col], ascending=[True, False], kind='mergesort')
          .copy()
    )

    results_dict = {}

    metrics_orig, top10_orig = _calculate_metrics(df_original)
    results_dict["Original"] = metrics_orig

    metrics_opt, _ = _calculate_metrics(df_optimistic)
    results_dict["Optimistic"] = metrics_opt

    metrics_pess, _ = _calculate_metrics(df_pessimistic)
    results_dict["Pessimistic"] = metrics_pess

    metrics_rand, _ = _calculate_metrics(df_random)
    results_dict["Random"] = metrics_rand

    if output_top10_file is not None:
        top10_orig.to_excel(output_top10_file, index=False)

    return {
        "metrics_by_strategy": results_dict,
        "global_stats": {
            "Num_CVEs": num_cves,
            "Total_Positive_Associations": total_positive,
        },
    }


if __name__ == "__main__":
    result = evaluate_ranking_metrics_baseline_style(
        excel_file="data/RankingEx/output/test_results/Ensemble_Weighted_Results_.xlsx",
        output_top10_file="data/RankingEx/output/test_results/top10_results_.xlsx",
        cve_col="CVE-ID",
        score_col="Final_Score",
        label_col="Label",
        topk_max=10,
    )

    strategies = ["Optimistic", "Original", "Random", "Pessimistic"]

    print("=" * 60)
    print(f"Num_CVEs: {result['global_stats']['Num_CVEs']}")
    print(f"Total_Positive_Associations: {result['global_stats']['Total_Positive_Associations']}")
    print("=" * 60)

    for k in [1, 3, 5, 10]:
        print(f"\n--- Metrics @ {k} (tie-break sensitivity) ---")
        for strategy in strategies:
            metrics = result['metrics_by_strategy'][strategy]
            print(
                f"[{strategy.ljust(11)}] "
                f"Precision@{k}: {metrics[f'Precision@{k}']:.4f} | "
                f"Recall@{k}: {metrics[f'Recall@{k}']:.4f} | "
                f"F1@{k}: {metrics[f'F1@{k}']:.4f}"
            )

    print("\n--- MRR (tie-break sensitivity) ---")
    for strategy in strategies:
        mrr = result['metrics_by_strategy'][strategy]['MRR']
        print(f"[{strategy.ljust(11)}] MRR: {mrr:.4f}")
    print("=" * 60)
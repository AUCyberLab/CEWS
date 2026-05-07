import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional


def load_scoring_df(
    file_path: str,
    sheet_name: Any = 0,
    cve_col: str = "CVE-ID",
    score_col: Optional[str] = "Final_Score",
    label_col: str = "Label",
    pred_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and clean a scoring file. The label column is required, and at least
    one of (score_col, pred_col) must be provided.
    """
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel.")

    if score_col is None and pred_col is None:
        raise ValueError("At least one of score_col or pred_col must be provided.")

    required_cols = {label_col}
    if score_col is not None:
        required_cols.add(score_col)
    if pred_col is not None:
        required_cols.add(pred_col)
    if cve_col is not None:
        required_cols.add(cve_col)

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    if score_col is not None:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    if pred_col is not None:
        df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")

    keep_cols = [label_col]
    if cve_col is not None:
        keep_cols.append(cve_col)
    if score_col is not None:
        keep_cols.append(score_col)
    if pred_col is not None:
        keep_cols.append(pred_col)

    df = df[keep_cols].dropna().copy()

    if df.empty:
        raise ValueError("Input data is empty after cleaning.")

    return df


def compute_scoring_metrics(
    df: pd.DataFrame,
    label_col: str = "Label",
    score_col: Optional[str] = "Final_Score",
    pred_col: Optional[str] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute classification metrics.
    - If score_col is given, compute MAE/MSE and binarize via threshold.
    - Otherwise use pred_col directly as the hard prediction.
    """
    y_true = df[label_col].astype(int).to_numpy()

    results = {}

    if score_col is not None:
        y_score = df[score_col].astype(float).to_numpy()
        results["MAE"] = float(np.mean(np.abs(y_score - y_true)))
        results["MSE"] = float(np.mean((y_score - y_true) ** 2))
        y_pred = (y_score >= threshold).astype(int)
    else:
        results["MAE"] = np.nan
        results["MSE"] = np.nan
        y_pred = df[pred_col].astype(int).to_numpy()

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results["Accuracy"] = float(accuracy)
    results["Precision"] = float(precision)
    results["Recall"] = float(recall)
    results["F1"] = float(f1)

    results["TP"] = tp
    results["TN"] = tn
    results["FP"] = fp
    results["FN"] = fn
    results["Num_Samples"] = int(len(y_true))

    return results


def bootstrap_scoring_ci(
    df: pd.DataFrame,
    label_col: str = "Label",
    score_col: Optional[str] = "Final_Score",
    pred_col: Optional[str] = None,
    threshold: float = 0.5,
    group_col: Optional[str] = "CVE-ID",
    n_bootstrap: int = 2000,
    ci: float = 95.0,
    random_seed: int = 42,
    block_bootstrap: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Bootstrap CI for scoring metrics.

    With block_bootstrap=True (recommended) the resampling unit is `group_col`
    (a CVE), so all rows belonging to the same CVE are resampled together.
    With block_bootstrap=False each row is sampled independently.
    """
    rng = np.random.default_rng(random_seed)

    point_estimates = compute_scoring_metrics(
        df=df,
        label_col=label_col,
        score_col=score_col,
        pred_col=pred_col,
        threshold=threshold,
    )

    metrics_to_collect = ["Accuracy", "Precision", "Recall", "F1"]
    if score_col is not None:
        metrics_to_collect = ["MAE", "MSE"] + metrics_to_collect

    bootstrap_values = {m: [] for m in metrics_to_collect}

    if block_bootstrap:
        if group_col is None:
            raise ValueError("group_col must be provided when block_bootstrap=True.")
        if group_col not in df.columns:
            raise ValueError(f"group_col '{group_col}' not found in dataframe.")

        groups = [g.copy() for _, g in df.groupby(group_col, sort=False)]
        n_groups = len(groups)
        if n_groups == 0:
            raise ValueError("No groups found for bootstrap.")

        for _ in range(n_bootstrap):
            sample_idx = rng.integers(low=0, high=n_groups, size=n_groups)
            sampled_df = pd.concat([groups[i] for i in sample_idx], ignore_index=True)

            sampled_metrics = compute_scoring_metrics(
                df=sampled_df,
                label_col=label_col,
                score_col=score_col,
                pred_col=pred_col,
                threshold=threshold,
            )

            for m in metrics_to_collect:
                bootstrap_values[m].append(sampled_metrics[m])
    else:
        n_rows = len(df)
        for _ in range(n_bootstrap):
            sample_idx = rng.integers(low=0, high=n_rows, size=n_rows)
            sampled_df = df.iloc[sample_idx].reset_index(drop=True)

            sampled_metrics = compute_scoring_metrics(
                df=sampled_df,
                label_col=label_col,
                score_col=score_col,
                pred_col=pred_col,
                threshold=threshold,
            )

            for m in metrics_to_collect:
                bootstrap_values[m].append(sampled_metrics[m])

    alpha = (100.0 - ci) / 2.0
    lower_q = alpha
    upper_q = 100.0 - alpha

    ci_results = {}
    for metric, values in bootstrap_values.items():
        ci_results[metric] = (
            float(np.percentile(values, lower_q)),
            float(np.percentile(values, upper_q)),
        )

    return point_estimates, ci_results


def print_scoring_results(
    point_estimates: Dict[str, float],
    ci_results: Dict[str, Tuple[float, float]],
    has_soft_score: bool = True,
) -> None:
    print(f"Num_Samples: {point_estimates['Num_Samples']}")
    print()

    if has_soft_score:
        for metric in ["MAE", "MSE"]:
            v = point_estimates[metric]
            lo, hi = ci_results[metric]
            print(f"{metric}: {v:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])")
        print()

    for metric in ["Accuracy", "Precision", "Recall", "F1"]:
        v = point_estimates[metric]
        lo, hi = ci_results[metric]
        print(f"{metric}: {v:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])")

    print()
    print(
        f"Confusion Matrix: TP={point_estimates['TP']}, TN={point_estimates['TN']}, "
        f"FP={point_estimates['FP']}, FN={point_estimates['FN']}"
    )


def run_scoring_ci_from_file(
    file_path: str,
    sheet_name: Any = 0,
    cve_col: str = "CVE-ID",
    score_col: Optional[str] = "Final_Score",
    pred_col: Optional[str] = None,
    label_col: str = "Label",
    threshold: float = 0.5,
    n_bootstrap: int = 2000,
    random_seed: int = 42,
    block_bootstrap: bool = True,
):
    df = load_scoring_df(
        file_path=file_path,
        sheet_name=sheet_name,
        cve_col=cve_col,
        score_col=score_col,
        label_col=label_col,
        pred_col=pred_col,
    )

    point_estimates, ci_results = bootstrap_scoring_ci(
        df=df,
        label_col=label_col,
        score_col=score_col,
        pred_col=pred_col,
        threshold=threshold,
        group_col=cve_col,
        n_bootstrap=n_bootstrap,
        ci=95.0,
        random_seed=random_seed,
        block_bootstrap=block_bootstrap,
    )

    print_scoring_results(
        point_estimates=point_estimates,
        ci_results=ci_results,
        has_soft_score=(score_col is not None),
    )

    return df, point_estimates, ci_results


if __name__ == "__main__":
    file_path = "data/Test_dataset/output/New_output/Ensemble_Weighted_Results_.xlsx"
    df, point_estimates, ci_results = run_scoring_ci_from_file(
        file_path=file_path,
        sheet_name=0,
        cve_col="CVE-ID",
        score_col="Final_Score",
        pred_col=None,
        label_col="Label",
        threshold=0.5,
        n_bootstrap=2000,
        random_seed=42,
        block_bootstrap=True,
    )

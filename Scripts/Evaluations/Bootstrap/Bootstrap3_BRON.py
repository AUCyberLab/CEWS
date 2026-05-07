import os
import ast
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def parse_to_set(val):
    """
    Parse a cell value into a set of stripped strings.

    Handles three input shapes:
      - NaN / 'none' / 'nan' / 'null' / empty   -> empty set
      - Python-list literal "['a', 'b']"        -> parsed via ast.literal_eval
      - Comma-separated string "a, b, c"        -> split on commas
    """
    if pd.isna(val) or str(val).strip().lower() in ['none', 'nan', 'null', '']:
        return set()

    val_str = str(val).strip()

    if val_str.startswith('[') and val_str.endswith(']'):
        try:
            parsed_list = ast.literal_eval(val_str)
            return set(str(x).strip() for x in parsed_list if str(x).strip())
        except Exception:
            pass

    elements = [x.strip() for x in val_str.split(',')]
    return set(x for x in elements if x and x.lower() != 'none')


def load_pairwise_df(
    input_file: str,
    candidate_col: str = 'CAPEC-ID',
    pred_col: str = 'related capec through cwe',
    label_col: str = 'Label',
    cve_col: Optional[str] = 'CVE-ID',
) -> pd.DataFrame:
    """Load pairwise data and derive Predicted / Actual columns."""
    if input_file.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(input_file)
    elif input_file.lower().endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel.")

    required_cols = [candidate_col, pred_col, label_col]
    if cve_col is not None:
        required_cols.append(cve_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()

    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(int)

    candidate_sets = df[candidate_col].apply(parse_to_set)
    pred_sets = df[pred_col].apply(parse_to_set)

    # Predicted = 1 iff the candidate CAPEC overlaps with the predicted CAPEC set
    df['Predicted'] = [
        1 if len(cand & pred) > 0 else 0
        for cand, pred in zip(candidate_sets, pred_sets)
    ]
    df['Actual'] = df[label_col].astype(int)

    return df


def compute_pairwise_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute hard-label metrics from a DataFrame already carrying Predicted / Actual."""
    y_pred = df['Predicted'].astype(int).to_numpy()
    y_true = df['Actual'].astype(int).to_numpy()

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total_samples = len(df)
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Num_Samples": int(total_samples),
    }


def bootstrap_pairwise_ci(
    df: pd.DataFrame,
    cve_col: Optional[str] = 'CVE-ID',
    n_bootstrap: int = 2000,
    ci: float = 95.0,
    random_seed: int = 42,
    block_bootstrap: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Bootstrap CI for pairwise hard-label metrics.

    With block_bootstrap=True (recommended) the resampling unit is `cve_col`,
    so all rows belonging to the same CVE are resampled together.
    """
    rng = np.random.default_rng(random_seed)

    point_estimates = compute_pairwise_metrics(df)

    metrics_to_collect = ["Accuracy", "Precision", "Recall", "F1"]
    bootstrap_values = {m: [] for m in metrics_to_collect}

    if block_bootstrap:
        if cve_col is None:
            raise ValueError("cve_col must be provided when block_bootstrap=True.")
        if cve_col not in df.columns:
            raise ValueError(f"cve_col '{cve_col}' not found in dataframe.")

        groups = [g.copy() for _, g in df.groupby(cve_col, sort=False)]
        n_groups = len(groups)

        if n_groups == 0:
            raise ValueError("No groups found for bootstrap.")

        for _ in range(n_bootstrap):
            sample_idx = rng.integers(low=0, high=n_groups, size=n_groups)
            sampled_df = pd.concat([groups[i] for i in sample_idx], ignore_index=True)

            sampled_metrics = compute_pairwise_metrics(sampled_df)
            for m in metrics_to_collect:
                bootstrap_values[m].append(sampled_metrics[m])
    else:
        n_rows = len(df)
        if n_rows == 0:
            raise ValueError("Empty dataframe for bootstrap.")

        for _ in range(n_bootstrap):
            sample_idx = rng.integers(low=0, high=n_rows, size=n_rows)
            sampled_df = df.iloc[sample_idx].reset_index(drop=True)

            sampled_metrics = compute_pairwise_metrics(sampled_df)
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


def save_pairwise_with_confusion(df: pd.DataFrame, output_file: str) -> None:
    """Save the DataFrame with per-row TP/TN/FP/FN flag columns."""
    df = df.copy()
    df['TP'] = ((df['Predicted'] == 1) & (df['Actual'] == 1)).astype(int)
    df['TN'] = ((df['Predicted'] == 0) & (df['Actual'] == 0)).astype(int)
    df['FP'] = ((df['Predicted'] == 1) & (df['Actual'] == 0)).astype(int)
    df['FN'] = ((df['Predicted'] == 0) & (df['Actual'] == 1)).astype(int)

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    df_out = df.drop(columns=['Predicted', 'Actual'])
    df_out.to_excel(output_file, index=False)


def print_pairwise_results(
    point_estimates: Dict[str, float],
    ci_results: Dict[str, Tuple[float, float]],
) -> None:
    print("=" * 60)
    print("Pairwise Evaluation with Bootstrap CI")
    print("=" * 60)
    print(f"Num_Samples: {point_estimates['Num_Samples']}")
    print(
        f"TP={point_estimates['TP']}, TN={point_estimates['TN']}, "
        f"FP={point_estimates['FP']}, FN={point_estimates['FN']}"
    )
    print("-" * 60)

    for metric in ["Accuracy", "Precision", "Recall", "F1"]:
        v = point_estimates[metric]
        lo, hi = ci_results[metric]
        print(f"{metric}: {v:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])")

    print("=" * 60)


def run_pairwise_ci_from_file(
    input_file: str,
    output_file: Optional[str] = None,
    candidate_col: str = 'CAPEC-ID',
    pred_col: str = 'related capec through cwe',
    label_col: str = 'Label',
    cve_col: Optional[str] = 'CVE-ID',
    n_bootstrap: int = 2000,
    random_seed: int = 42,
    block_bootstrap: bool = True,
):
    df = load_pairwise_df(
        input_file=input_file,
        candidate_col=candidate_col,
        pred_col=pred_col,
        label_col=label_col,
        cve_col=cve_col,
    )

    point_estimates, ci_results = bootstrap_pairwise_ci(
        df=df,
        cve_col=cve_col,
        n_bootstrap=n_bootstrap,
        ci=95.0,
        random_seed=random_seed,
        block_bootstrap=block_bootstrap,
    )

    print_pairwise_results(point_estimates, ci_results)

    if output_file is not None:
        save_pairwise_with_confusion(df, output_file)
        print(f"Per-row evaluation file saved to: {output_file}")

    return df, point_estimates, ci_results


if __name__ == "__main__":
    MAPPED_FILE = "data/Test_dataset/BRON_cves_mapped.xlsx"
    EVALUATED_OUTPUT = "data/Test_dataset/BRON_cves_evaluated_with_labels.xlsx"

    df, point_estimates, ci_results = run_pairwise_ci_from_file(
        input_file=MAPPED_FILE,
        output_file=EVALUATED_OUTPUT,
        candidate_col='CAPEC-ID',
        pred_col='related capec through cwe',
        label_col='Label',
        cve_col='CVE-ID',
        n_bootstrap=2000,
        random_seed=42,
        block_bootstrap=True,
    )
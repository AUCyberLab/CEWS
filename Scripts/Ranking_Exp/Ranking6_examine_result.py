import os
import pandas as pd
import numpy as np


def calculate_ensemble_scores(
    input_filepath,
    output_dir,
    threshold=0.5,
    masked_features=None,
    weights=None,
):
    """
    Ensemble per-feature match scores via majority Vote and Weighted Average.
    Supports masking out features for ablation studies.

    Args:
        input_filepath:   Path to the merged scores Excel file.
        output_dir:       Directory to write result files into.
        threshold:        Cutoff used to binarize the weighted-average score.
        masked_features:  Feature column names to exclude from the ensemble.
        weights:          {column: weight} dict. Unspecified columns default to 1.0.
    """
    if masked_features is None:
        masked_features = []
    if weights is None:
        weights = {}

    # All candidate score columns. Note: Structure uses a space, not an underscore.
    score_cols = [
        'Prerequisite_Match_Score',
        'Impact_Match_Score',
        'Mechanism_Match_Score',
        'Description_Match_Score',
        'Structure Score',
    ]

    active_cols = [col for col in score_cols if col not in masked_features]

    print("\n" + "=" * 60)
    print(f"Ensemble scoring (Weighted Average threshold: {threshold})")

    if not active_cols:
        print("[Error] All features are masked, nothing to compute.")
        return

    active_weights = [weights.get(col, 1.0) for col in active_cols]

    print(f"Active features ({len(active_cols)}/{len(score_cols)}):")
    for col, w in zip(active_cols, active_weights):
        print(f"  [+] {col} (weight: {w})")

    if masked_features:
        print("Masked (ablated) features:")
        for col in masked_features:
            print(f"  [-] {col}")
    print("=" * 60)

    def calc_metrics(y_true, y_pred):
        TP = ((y_pred == 1) & (y_true == 1)).sum()
        FP = ((y_pred == 1) & (y_true == 0)).sum()
        TN = ((y_pred == 0) & (y_true == 0)).sum()
        FN = ((y_pred == 0) & (y_true == 1)).sum()

        accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return accuracy, precision, recall, f1

    try:
        df = pd.read_excel(input_filepath)
    except Exception as e:
        print(f"Failed to read input file: {e}")
        return

    missing_cols = [col for col in score_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing score columns; check the upstream merge step: {missing_cols}")
        return

    if 'Label' not in df.columns:
        print("Label column not found; cannot run comparative analysis.")
        return

    df['Label'] = pd.to_numeric(df['Label'], errors='coerce')
    df = df.dropna(subset=['Label']).copy()
    y_true = df['Label'].astype(int)

    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    os.makedirs(output_dir, exist_ok=True)

    # Build a filename suffix that records which features were ablated
    file_suffix = ""
    if masked_features:
        short_names = [f.split('_')[0].split(' ')[0].lower() for f in masked_features]
        file_suffix = "_no_" + "_".join(short_names)

    # ==========================================
    # Strategy 1: Majority Vote (weights are ignored here, only masking applies)
    # ==========================================
    df_vote = df.copy()

    # The Structure score is a continuous value in [0, 1]; collapse strict-interior
    # values to 0.5 so it behaves like a discrete vote in {0, 0.5, 1}, matching
    # the value space the other score columns produce.
    structure_col = 'Structure Score'
    if structure_col in active_cols:
        mask_structure = (df_vote[structure_col] > 0) & (df_vote[structure_col] < 1)
        df_vote.loc[mask_structure, structure_col] = 0.5

    # Row-wise mode across the active feature columns
    modes = df_vote[active_cols].mode(axis=1, dropna=True)
    df_vote['Final_Score'] = modes[0]

    df_vote['Pred_0.5_as_0'] = np.where(df_vote['Final_Score'] >= 1, 1, 0)
    df_vote['Pred_0.5_as_1'] = np.where(df_vote['Final_Score'] >= 0.5, 1, 0)

    vote_acc_0, vote_prec_0, vote_rec_0, vote_f1_0 = calc_metrics(y_true, df_vote['Pred_0.5_as_0'])
    vote_acc_1, vote_prec_1, vote_rec_1, vote_f1_1 = calc_metrics(y_true, df_vote['Pred_0.5_as_1'])

    vote_output = os.path.join(output_dir, f'Ensemble_Vote_Results_{file_suffix}.xlsx')
    df_vote.to_excel(vote_output, index=False)

    # ==========================================
    # Strategy 2: Weighted Average
    # ==========================================
    df_weighted = df.copy()

    # numpy.average normalizes the weights internally, so passing un-normalized
    # weights is fine.
    df_weighted['Final_Score'] = np.average(df_weighted[active_cols], axis=1, weights=active_weights)

    mae = (df_weighted['Final_Score'] - y_true).abs().mean()
    mse = ((df_weighted['Final_Score'] - y_true) ** 2).mean()

    df_weighted['Predicted'] = (df_weighted['Final_Score'] >= threshold).astype(int)

    weighted_acc, weighted_prec, weighted_rec, weighted_f1 = calc_metrics(y_true, df_weighted['Predicted'])

    weighted_output = os.path.join(output_dir, f'Ensemble_Weighted_Results_{file_suffix}.xlsx')
    df_weighted.to_excel(weighted_output, index=False)

    # ==========================================
    # Summary
    # ==========================================
    print("\n[Strategy 1: Vote]")
    print(" Variant A: treat 0.5 as 0 (conservative)")
    print(f"  - Accuracy: {vote_acc_0:.4f} | Precision: {vote_prec_0:.4f} | Recall: {vote_rec_0:.4f} | F1: {vote_f1_0:.4f}")

    print("\n Variant B: treat 0.5 as 1 (sensitive)")
    print(f"  - Accuracy: {vote_acc_1:.4f} | Precision: {vote_prec_1:.4f} | Recall: {vote_rec_1:.4f} | F1: {vote_f1_1:.4f}")

    print("\n[Strategy 2: Weighted Average]")
    print(" Continuous-score error (no binarization):")
    print(f"  - MAE: {mae:.4f} | MSE: {mse:.4f}")
    print(f" Classification metrics (threshold >= {threshold}):")
    print(f"  - Accuracy: {weighted_acc:.4f} | Precision: {weighted_prec:.4f} | Recall: {weighted_rec:.4f} | F1: {weighted_f1:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    input_file = 'data/RankingEx/output/TL_predictions_5_score_with_label.xlsx'
    output_directory = 'data/RankingEx/output/test_results/'

    # Per-feature weights (un-normalized; numpy.average normalizes internally)
    custom_weights = {
        'Mechanism_Match_Score': 1.0,
        'Impact_Match_Score': 1.0,
        'Prerequisite_Match_Score': 0.5,
        'Description_Match_Score': 2.5,
        'Structure Score': 1.5
    }

    # Experiment 1: baseline (all 5 features, weighted)
    calculate_ensemble_scores(
        input_filepath=input_file,
        output_dir=output_directory,
        threshold=0.5,
        masked_features=[],
        weights=custom_weights,
    )

    # Experiment 2: ablation example (mask features; remaining weights re-normalize)
    # calculate_ensemble_scores(
    #     input_filepath=input_file,
    #     output_dir=output_directory,
    #     threshold=0.6,
    #     masked_features=[
    #         'Prerequisite_Match_Score',
    #         'Impact_Match_Score',
    #         'Mechanism_Match_Score',
    #         'Structure Score',
    #     ],
    #     weights=custom_weights,
    # )
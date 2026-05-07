import pandas as pd
import numpy as np
import os
import itertools


def search_best_ensemble_weights(val_filepath, output_dir, threshold=0.5):
    """
    Search for the best Weighted Average weight combination on the validation set.
    Outputs single-feature baselines, an equal-weights baseline, and the Top 3
    best weight combinations.
    """
    print("\n" + "=" * 70)
    print(f"Searching for best weight combination (val set: {os.path.basename(val_filepath)} | threshold: {threshold})")
    print("=" * 70)

    score_cols = [
        'Prerequisite_Match_Score',
        'Impact_Match_Score',
        'Mechanism_Match_Score',
        'Description_Match_Score',
        'Structure Score'
    ]

    def calc_metrics(y_true, y_pred):
        TP = ((y_pred == 1) & (y_true == 1)).sum()
        FP = ((y_pred == 1) & (y_true == 0)).sum()
        TN = ((y_pred == 0) & (y_true == 0)).sum()
        FN = ((y_pred == 0) & (y_true == 1)).sum()

        accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return accuracy, precision, recall, f1_score

    try:
        df = pd.read_excel(val_filepath)
    except Exception as e:
        print(f"Failed to load validation set: {e}")
        return

    missing_cols = [col for col in score_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing score columns: {missing_cols}")
        return

    if 'Label' not in df.columns:
        print("'Label' column not found; cannot run comparison.")
        return

    df['Label'] = pd.to_numeric(df['Label'], errors='coerce')
    df = df.dropna(subset=['Label']).copy()
    y_true = df['Label'].astype(int)

    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    os.makedirs(output_dir, exist_ok=True)
    # Convert to numpy array to speed up downstream computation
    features_data = df[score_cols].values

    # ==========================================
    # Stage 1: Baselines (single feature and equal weights)
    # ==========================================
    print("\n[Stage 1: Baselines]")

    print("--- Single-feature performance ---")
    for i, col in enumerate(score_cols):
        scores = features_data[:, i]
        preds = (scores >= threshold).astype(int)
        acc, prec, rec, f1 = calc_metrics(y_true, preds)
        mae = np.abs(scores - y_true).mean()
        print(f"{col:25s} | F1: {f1:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | MAE: {mae:.4f}")

    equal_weights = np.ones(5)
    equal_scores = np.average(features_data, axis=1, weights=equal_weights)
    eq_preds = (equal_scores >= threshold).astype(int)
    eq_acc, eq_prec, eq_rec, eq_f1 = calc_metrics(y_true, eq_preds)
    eq_mae = np.abs(equal_scores - y_true).mean()
    print("\n--- Equal Weights (1.0 each) ---")
    print(f"Average                   | F1: {eq_f1:.4f} | Acc: {eq_acc:.4f} | Prec: {eq_prec:.4f} | Rec: {eq_rec:.4f} | MAE: {eq_mae:.4f}")

    # ==========================================
    # Stage 2: Grid Search for best weights
    # ==========================================
    # Description and Impact get a wider candidate range, but capped at 3.0 to keep ratios reasonable.
    weight_space = {
        'Prerequisite_Match_Score': [0.5, 1.0, 1.5],
        'Impact_Match_Score':       [1.0, 1.5, 2.0, 2.5],
        'Mechanism_Match_Score':    [0.5, 1.0, 1.5],
        'Description_Match_Score':  [1.0, 1.5, 2.0, 2.5, 3.0],
        'Structure Score':          [0.5, 1.0, 1.5]
    }

    keys = list(weight_space.keys())
    all_combinations = list(itertools.product(*[weight_space[k] for k in keys]))

    results = []

    for combo in all_combinations:
        weights = np.array(combo)

        w_scores = np.average(features_data, axis=1, weights=weights)
        w_preds = (w_scores >= threshold).astype(int)

        acc, prec, rec, f1 = calc_metrics(y_true, w_preds)
        mae = np.abs(w_scores - y_true).mean()
        mse = ((w_scores - y_true) ** 2).mean()

        results.append({
            'weights': dict(zip(keys, combo)),
            'F1': f1,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'MAE': mae,
            'MSE': mse
        })

    # Sort by F1 desc, then Accuracy desc, then MAE asc
    results.sort(key=lambda x: (x['F1'], x['Accuracy'], -x['MAE']), reverse=True)

    # ==========================================
    # Stage 3: Print and save top results
    # ==========================================
    print("\n[Stage 2: Top 3 Weight Combinations (Sorted by F1-Score)]")

    for i in range(min(3, len(results))):
        res = results[i]
        w_dict = res['weights']
        print(f"\nTop {i+1}:")
        print(f"  F1: {res['F1']:.4f} | Acc: {res['Accuracy']:.4f} | Prec: {res['Precision']:.4f} | Rec: {res['Recall']:.4f} | MAE: {res['MAE']:.4f}")
        print("  Weights:")
        for k, v in w_dict.items():
            print(f"    - {k}: {v}")

    # Persist the Top 1 predictions for downstream evaluation
    top1 = results[0]
    top1_weights = np.array([top1['weights'][k] for k in score_cols])
    df['Final_Score'] = np.average(features_data, axis=1, weights=top1_weights)
    df['Predicted'] = (df['Final_Score'] >= threshold).astype(int)

    output_filepath = os.path.join(output_dir, "Best_Weighted_Ensemble_ValSet.xlsx")
    df.to_excel(output_filepath, index=False)
    print("\n" + "=" * 70)
    print(f"Top 1 weighted predictions saved to: {output_filepath}")
    print("=" * 70)

if __name__ == "__main__":
    val_file = 'data/Test_dataset/output/Test_output/New_set_5_scores_val.xlsx'
    output_directory = 'data/Test_dataset/output/Ensemble_Results'

    search_best_ensemble_weights(
        val_filepath=val_file,
        output_dir=output_directory,
        threshold=0.5
    )
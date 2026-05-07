import pandas as pd
import os


def evaluate_predictions(file_path):
    print(f"==================================================")
    print(f"Start evaluating prediction file: {file_path}")
    print(f"==================================================")

    # 1. Load data
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return

    # Choose reader based on file extension
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    # 2. Data cleaning and preprocessing
    score_col = 'Match_Score'
    label_col = 'Label'

    if score_col not in df.columns or label_col not in df.columns:
        print(f"[Error] The data must contain both '{score_col}' and '{label_col}' columns.")
        print(f"Current columns: {list(df.columns)}")
        return

    # Coerce scores to numeric, keeping floats (e.g. 0.5); failed parses (NaN) are treated as 0.0
    df[score_col] = pd.to_numeric(df[score_col], errors='coerce').fillna(0.0)
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')

    # Drop rows with missing ground-truth labels
    df = df.dropna(subset=[label_col])

    # Convert ground-truth labels to 0/1 integers
    df['Actual'] = df[label_col].astype(int)

    # 3. Core logic: decide whether the model predicts True (1) or False (0) for classification metrics
    # Note: this binarized Predicted column is only used for accuracy/recall/etc.
    df['Predicted'] = (df[score_col] != 0).astype(int)

    # 4. Compute MAE and MSE (continuous/regression metrics)
    # Use the raw Match_Score (which can be 0.5, etc.) minus the Actual label
    mae = (df[score_col] - df['Actual']).abs().mean()
    mse = ((df[score_col] - df['Actual']) ** 2).mean()

    # 5. Compute base counts for classification metrics: TP, FP, TN, FN
    TP = ((df['Predicted'] == 1) & (df['Actual'] == 1)).sum()
    FP = ((df['Predicted'] == 1) & (df['Actual'] == 0)).sum()
    TN = ((df['Predicted'] == 0) & (df['Actual'] == 0)).sum()
    FN = ((df['Predicted'] == 0) & (df['Actual'] == 1)).sum()

    total_samples = len(df)
    correct_predictions = TP + TN

    # 6. Compute downstream evaluation metrics
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 7. Print results
    print(f"Total samples     : {total_samples}")
    print(f"Correct predictions: {correct_predictions}\n")

    print(f"--- Confusion Matrix ---")
    print(f"TP (predicted positive, actually positive): {TP}")
    print(f"FP (predicted positive, actually negative): {FP}  <-- false alarm (model too sensitive)")
    print(f"TN (predicted negative, actually negative): {TN}")
    print(f"FN (predicted negative, actually positive): {FN}  <-- miss (model failed to detect a match)\n")

    print(f"--- Core Metrics ---")
    print(f"Accuracy : {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall   : {recall:.2%}")
    print(f"F1 Score : {f1_score:.4f}\n")

    print(f"--- Error Metrics ---")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"MSE (Mean Squared Error) : {mse:.4f}")
    print(f"==================================================")

    # Write the evaluation results back to an Excel file
    df.to_excel(file_path.replace('.xlsx', '_evaluated.xlsx'), index=False)

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1_score,
        "MAE": mae, "MSE": mse
    }


if __name__ == "__main__":
    test_file_path = "data/Test_dataset/output/Test_output/Prerequisite_predictions_newtest_b16_with_label.xlsx"
    evaluate_predictions(test_file_path)
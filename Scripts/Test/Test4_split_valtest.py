import pandas as pd
import os
from sklearn.model_selection import train_test_split


def split_dataset_by_cve(file_path, id_column='CVE-ID', val_ratio=0.4, random_state=42):
    """
    Split an Excel dataset into Validation (40%) and Test (60%) sets
    based on unique CVE-IDs.
    """
    print("=" * 50)
    print(f"Start processing file: {file_path}")

    # 1. Check whether the file exists
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return

    # 2. Read the Excel file
    df = pd.read_excel(file_path)

    # 3. Check that the specified column exists
    if id_column not in df.columns:
        print(f"[Error] The data must contain the '{id_column}' column. Current columns: {list(df.columns)}")
        return

    # 4. Extract unique CVE-IDs and split them
    unique_cves = df[id_column].unique()
    total_cves = len(unique_cves)
    print(f"The dataset contains {total_cves} unique {id_column} values.")

    # Split unique IDs into 40% validation and 60% test
    val_cves, test_cves = train_test_split(
        unique_cves,
        train_size=val_ratio,
        test_size=1.0 - val_ratio,
        random_state=random_state
    )

    # 5. Pull the corresponding rows for each split
    val_df = df[df[id_column].isin(val_cves)]
    test_df = df[df[id_column].isin(test_cves)]

    # 6. Build output file names (append _val and _test to the original name)
    file_dir, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)

    val_file_path = os.path.join(file_dir, f"{base_name}_val{ext}")
    test_file_path = os.path.join(file_dir, f"{base_name}_test{ext}")

    # 7. Save to new Excel files
    print("Saving validation set...")
    val_df.to_excel(val_file_path, index=False)

    print("Saving test set...")
    test_df.to_excel(test_file_path, index=False)

    # 8. Print summary
    print("-" * 50)
    print("Split complete. Summary:")
    print(f"Original total rows: {len(df)}")
    print(f"[Validation set] {len(val_cves)} unique CVEs, {len(val_df)} rows -> {val_file_path}")
    print(f"[Test set]       {len(test_cves)} unique CVEs, {len(test_df)} rows -> {test_file_path}")
    print("=" * 50)

if __name__ == "__main__":
    input_file = "data/Test_dataset/output/New_output/New_set_5_scores.xlsx"
    split_dataset_by_cve(file_path=input_file, id_column='CVE-ID', val_ratio=0.5)
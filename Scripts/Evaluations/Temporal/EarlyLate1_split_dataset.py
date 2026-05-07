import re
import pandas as pd


def split_excel_by_cve_year(input_excel, train_output_excel, test_output_excel):
    """
    Split an Excel file into train/test sets based on the year embedded in CVE-ID:
      - 2022 and earlier -> training set
      - 2023-2024        -> test set

    Args:
        input_excel:        Path to the source Excel file.
        train_output_excel: Path for the training-set output.
        test_output_excel:  Path for the test-set output.
    """
    df = pd.read_excel(input_excel)

    if "CVE-ID" not in df.columns:
        raise ValueError("Column 'CVE-ID' not found in the Excel file.")

    def extract_year(cve_id):
        """Pull the 4-digit year out of an ID like 'CVE-2021-34527'."""
        if pd.isna(cve_id):
            return None
        match = re.match(r"^CVE-(\d{4})-\d+$", str(cve_id).strip(), re.IGNORECASE)
        return int(match.group(1)) if match else None

    df["CVE_Year"] = df["CVE-ID"].apply(extract_year)

    invalid_rows = df[df["CVE_Year"].isna()]
    if not invalid_rows.empty:
        print(
            f"Warning: {len(invalid_rows)} rows have an unparseable CVE-ID "
            "and will be excluded from both train and test sets."
        )

    train_df = df[df["CVE_Year"].notna() & (df["CVE_Year"] <= 2022)].copy()
    test_df = df[df["CVE_Year"].between(2023, 2024, inclusive="both")].copy()

    # Drop the helper column from the output
    train_df.drop(columns=["CVE_Year"], inplace=True)
    test_df.drop(columns=["CVE_Year"], inplace=True)

    train_df.to_excel(train_output_excel, index=False)
    test_df.to_excel(test_output_excel, index=False)

    print(f"Training set saved to: {train_output_excel} ({len(train_df)} rows)")
    print(f"Test set saved to:     {test_output_excel} ({len(test_df)} rows)")


if __name__ == "__main__":
    split_excel_by_cve_year(
        input_excel="data/Description/Processed_Data/Train_SFT_Samples_des_all_OOD_42.xlsx",
        train_output_excel="data/Early_late_dataset/Description_train.xlsx",
        test_output_excel="data/Early_late_dataset/Description_test.xlsx",
    )
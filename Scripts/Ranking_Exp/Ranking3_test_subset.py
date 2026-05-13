import os
import math
import pandas as pd


def split_excel_by_cve(
    input_excel: str,
    output_dir: str,
    cve_col: str = "CVE-ID",
    num_subsets: int = 10,
):
    """
    Split an Excel dataset into N subsets along CVE-ID boundaries so every row
    belonging to the same CVE stays together. Each subset is saved as its own xlsx.

    Args:
        input_excel: Path to the input Excel file.
        output_dir:  Directory where subset files will be written.
        cve_col:     Column holding the CVE identifier.
        num_subsets: Number of subsets to produce.
    """
    if not os.path.exists(input_excel):
        raise FileNotFoundError(f"Input file not found: {input_excel}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"[System] Reading: {input_excel}")
    df = pd.read_excel(input_excel)

    if cve_col not in df.columns:
        raise ValueError(f"Column '{cve_col}' not found. Available columns: {list(df.columns)}")

    # Drop rows with a missing CVE-ID
    df = df[df[cve_col].notna()].copy()

    # Preserve the original order of unique CVEs
    unique_cves = df[cve_col].drop_duplicates().tolist()
    total_cves = len(unique_cves)

    if total_cves == 0:
        raise ValueError("No valid CVE-IDs found.")

    print(f"[System] Total unique CVEs: {total_cves}")
    print(f"[System] Splitting into {num_subsets} subsets...")

    cves_per_subset = math.ceil(total_cves / num_subsets)

    summary = []

    for subset_idx in range(num_subsets):
        start = subset_idx * cves_per_subset
        end = min((subset_idx + 1) * cves_per_subset, total_cves)

        subset_cves = unique_cves[start:end]

        # Once we run out of CVEs, stop emitting empty files
        if not subset_cves:
            break

        subset_df = df[df[cve_col].isin(subset_cves)].copy()

        output_file = os.path.join(
            output_dir, f"TL_CVE_CAPEC_cross_dataset_{subset_idx + 1:02d}.xlsx"
        )
        subset_df.to_excel(output_file, index=False)

        summary.append({
            "subset": f"subset_{subset_idx + 1:02d}",
            "unique_cves": len(subset_cves),
            "rows": len(subset_df),
            "output_file": output_file,
        })

        print(
            f"[Saved] subset_{subset_idx + 1:02d}: "
            f"{len(subset_cves)} CVEs, {len(subset_df)} rows -> {output_file}"
        )

    summary_df = pd.DataFrame(summary)
    summary_file = os.path.join(output_dir, "split_summary.xlsx")
    summary_df.to_excel(summary_file, index=False)

    print("\n[System] Done.")
    print(f"[System] Summary saved to: {summary_file}")


if __name__ == "__main__":
    split_excel_by_cve(
        input_excel="data/RankingEx/TL_CVE_CAPEC_cross_dataset.xlsx",
        output_dir="data/RankingEx/TL_CVE_CAPEC_cross_dataset_subset",
        cve_col="CVE-ID",
        num_subsets=10,
    )
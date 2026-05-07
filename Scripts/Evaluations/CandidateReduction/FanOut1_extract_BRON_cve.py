import re
import ast
from pathlib import Path
import pandas as pd


def read_table(file_path):
    """Read an Excel or CSV table; for CSVs, try several encodings before giving up."""
    file_path = Path(file_path)

    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif file_path.suffix.lower() == ".csv":
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1", "gbk"]
        last_error = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=enc, low_memory=False)
                break
            except UnicodeDecodeError as e:
                last_error = e
        else:
            raise ValueError(f"Unable to read {file_path}. Last error: {last_error}")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Drop unnamed/empty columns from prior writes and trim header whitespace
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]

    return df


def write_table(df, file_path):
    file_path = Path(file_path)

    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        df.to_excel(file_path, index=False)
    elif file_path.suffix.lower() == ".csv":
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def extract_valid_cwes(x):
    """
    Keep only entries shaped like 'CWE-<digits>'.

    Tolerates several input forms:
      - ['CWE-266', 'CWE-269']
      - []
      - unknown
      - ['unknown']
      - nan
    """
    if pd.isna(x):
        return []

    if isinstance(x, list):
        items = x
    else:
        s = str(x).strip()

        if s == "" or s.lower() in {"nan", "none", "null", "unknown", "[]"}:
            return []

        try:
            parsed = ast.literal_eval(s)
            items = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            items = [s]

    return [str(item).strip() for item in items if re.fullmatch(r"CWE-\d+", str(item).strip())]


def merge_unique_cwes(series_of_lists):
    """
    Merge several CWE lists for the same CVE into a single deduplicated string.

    e.g. ['CWE-266', 'CWE-269'] + ['CWE-269', 'CWE-842']
         -> 'CWE-266, CWE-269, CWE-842'
    """
    merged = []
    seen = set()

    for cwe_list in series_of_lists:
        if not isinstance(cwe_list, list):
            continue
        for cwe in cwe_list:
            if cwe not in seen:
                seen.add(cwe)
                merged.append(cwe)

    return ", ".join(merged)


def filter_a_by_b_valid_cwe_with_relatedcwe(
    file_a,
    file_b,
    output_file,
    cve_col_a="CVE-ID",
    cve_col_b="CVE-ID",
    cwe_col_b="Related-CWE",
    output_relatedcwe_col="relatedcwe",
):
    """
    Keep only rows in A whose CVE-ID has at least one valid CWE in B, and add
    a `relatedcwe` column to A built from B's CWE entries (deduped, joined).
    """
    df_a = read_table(file_a)
    df_b = read_table(file_b)

    if cve_col_a not in df_a.columns:
        raise KeyError(f"Column not found in file A: {cve_col_a}")
    if cve_col_b not in df_b.columns:
        raise KeyError(f"Column not found in file B: {cve_col_b}")
    if cwe_col_b not in df_b.columns:
        raise KeyError(f"Column not found in file B: {cwe_col_b}")

    df_a[cve_col_a] = df_a[cve_col_a].astype(str).str.strip()
    df_b[cve_col_b] = df_b[cve_col_b].astype(str).str.strip()

    df_b["_valid_cwe_list"] = df_b[cwe_col_b].apply(extract_valid_cwes)

    # Drop B rows that have no recognizable CWE so they don't pull empty values into the merge
    df_b_valid = df_b[df_b["_valid_cwe_list"].apply(lambda x: len(x) > 0)].copy()

    # Aggregate every B row of the same CVE into one merged CWE string
    cve_to_cwes = (
        df_b_valid.groupby(cve_col_b)["_valid_cwe_list"]
        .apply(merge_unique_cwes)
        .reset_index()
        .rename(columns={"_valid_cwe_list": output_relatedcwe_col})
    )

    filtered_a = df_a[df_a[cve_col_a].isin(cve_to_cwes[cve_col_b])].copy()
    filtered_a = filtered_a.merge(
        cve_to_cwes,
        left_on=cve_col_a,
        right_on=cve_col_b,
        how="left",
    )

    # Drop the duplicate CVE column produced by the merge when A and B use different names
    if cve_col_b in filtered_a.columns and cve_col_b != cve_col_a:
        filtered_a = filtered_a.drop(columns=[cve_col_b])

    write_table(filtered_a, output_file)

    print("========== Summary ==========")
    print(f"A total rows:                  {len(df_a)}")
    print(f"A unique CVEs:                 {df_a[cve_col_a].nunique()}")
    print(f"Unique CVEs with valid CWE:    {cve_to_cwes[cve_col_b].nunique()}")
    print(f"Filtered A total rows:         {len(filtered_a)}")
    print(f"Filtered A unique CVEs:        {filtered_a[cve_col_a].nunique()}")

    return filtered_a


if __name__ == "__main__":
    filter_a_by_b_valid_cwe_with_relatedcwe(
        file_a="data/Fan-out-reduction/Structure_prediction_TL.xlsx",
        file_b="data/Fan-out-reduction/CVE_CWE_withpageinfo_all.csv",
        output_file="data/Fan-out-reduction/Structure_prediction_TL_BRON.xlsx",
        cve_col_a="CVE ID",
        cve_col_b="CVE-ID",
        cwe_col_b="CWE-ID",
    )
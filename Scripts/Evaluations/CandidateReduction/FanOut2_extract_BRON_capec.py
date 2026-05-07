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


def extract_cwe_ids(x):
    """
    Extract CWE-IDs from a relatedcwe text field.

    Handles 'CWE-266, CWE-269', "['CWE-266', 'CWE-269']", nan, 'unknown', etc.
    """
    if pd.isna(x):
        return []

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "unknown", "[]"}:
        return []

    return re.findall(r"CWE-\d+", s)


def parse_capec_list(x):
    """
    Parse the Related_CAPEC_IDs column from a CWE-CAPEC mapping file.

    Accepts [21, 59], 'CAPEC-21', '21', strings with embedded numbers, etc.
    Always returns a deduplicated list of canonical 'CAPEC-N' strings.
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

    capecs = []
    for item in items:
        item_str = str(item).strip()

        if item_str == "" or item_str.lower() in {"nan", "none", "null", "unknown"}:
            continue

        if re.fullmatch(r"CAPEC-\d+", item_str):
            capecs.append(item_str)
        elif re.fullmatch(r"\d+", item_str):
            capecs.append(f"CAPEC-{item_str}")
        else:
            m = re.search(r"(\d+)", item_str)
            if m:
                capecs.append(f"CAPEC-{m.group(1)}")

    # Order-preserving deduplication
    seen = set()
    result = []
    for c in capecs:
        if c not in seen:
            seen.add(c)
            result.append(c)

    return result


def merge_unique_lists(series_of_lists):
    """Merge multiple lists, preserving first-seen order."""
    seen = set()
    merged = []
    for lst in series_of_lists:
        if not isinstance(lst, list):
            continue
        for item in lst:
            if item not in seen:
                seen.add(item)
                merged.append(item)
    return merged


def add_relatedcapec_from_cwe_capec_keep_first_row(
    input_file,
    cwe_capec_file,
    output_file,
    cve_col="CVE-ID",
    relatedcwe_col="relatedcwe",
    cwe_capec_cwe_col="CWE-ID",
    cwe_capec_capec_col="Related_CAPEC_IDs",
    output_relatedcapec_col="relatedcapec",
    output_relatedcapec_count_col="relatedcapec_count",
):
    """
    Walk every row's relatedcwe through the CWE-CAPEC mapping table and add:
      1. relatedcapec        - comma-joined string of related CAPEC-IDs
      2. relatedcapec_count  - count of those CAPECs

    Then keep only rows with at least one related CAPEC, and dedupe so each
    CVE-ID contributes a single row (first occurrence).
    """
    df_input = read_table(input_file)
    df_map = read_table(cwe_capec_file)

    if cve_col not in df_input.columns:
        raise KeyError(f"Column not found in input file: {cve_col}")
    if relatedcwe_col not in df_input.columns:
        raise KeyError(f"Column not found in input file: {relatedcwe_col}")
    if cwe_capec_cwe_col not in df_map.columns:
        raise KeyError(f"Column not found in CWE-CAPEC file: {cwe_capec_cwe_col}")
    if cwe_capec_capec_col not in df_map.columns:
        raise KeyError(f"Column not found in CWE-CAPEC file: {cwe_capec_capec_col}")

    df_input[cve_col] = df_input[cve_col].astype(str).str.strip()
    df_map[cwe_capec_cwe_col] = df_map[cwe_capec_cwe_col].astype(str).str.strip()

    df_input["_relatedcwe_list"] = df_input[relatedcwe_col].apply(extract_cwe_ids)
    df_map["_relatedcapec_list"] = df_map[cwe_capec_capec_col].apply(parse_capec_list)

    # Same CWE may appear in multiple rows of the mapping table; merge their CAPECs
    cwe_to_capecs_df = (
        df_map.groupby(cwe_capec_cwe_col)["_relatedcapec_list"]
        .apply(merge_unique_lists)
        .reset_index()
    )

    cwe_to_capecs = dict(
        zip(cwe_to_capecs_df[cwe_capec_cwe_col], cwe_to_capecs_df["_relatedcapec_list"])
    )

    def collect_related_capecs(cwe_list):
        if not isinstance(cwe_list, list) or len(cwe_list) == 0:
            return []

        seen = set()
        merged = []
        for cwe in cwe_list:
            for capec in cwe_to_capecs.get(cwe, []):
                if capec not in seen:
                    seen.add(capec)
                    merged.append(capec)
        return merged

    df_input["_relatedcapec_list"] = df_input["_relatedcwe_list"].apply(collect_related_capecs)

    df_input[output_relatedcapec_col] = df_input["_relatedcapec_list"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) and len(x) > 0 else ""
    )
    df_input[output_relatedcapec_count_col] = df_input["_relatedcapec_list"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    # Keep only rows with at least one related CAPEC
    filtered_df = df_input[df_input[output_relatedcapec_count_col] > 0].copy()

    # Keep first occurrence per CVE-ID
    filtered_df = filtered_df.drop_duplicates(subset=[cve_col], keep="first").copy()

    filtered_df = filtered_df.drop(
        columns=["_relatedcwe_list", "_relatedcapec_list"],
        errors="ignore",
    )

    write_table(filtered_df, output_file)

    print("========== Summary ==========")
    print(f"Input total rows:          {len(df_input)}")
    print(f"Input unique CVEs:         {df_input[cve_col].nunique()}")
    print(f"Filtered total rows:       {len(filtered_df)}")
    print(f"Filtered unique CVEs:      {filtered_df[cve_col].nunique()}")
    print()
    print(filtered_df[[
        cve_col,
        relatedcwe_col,
        output_relatedcapec_col,
        output_relatedcapec_count_col,
    ]].head(10))

    return filtered_df


if __name__ == "__main__":
    add_relatedcapec_from_cwe_capec_keep_first_row(
        input_file="data/Fan-out-reduction/Structure_prediction_TL_BRON.xlsx",
        cwe_capec_file="data/Fan-out-reduction/CWE_parsed_view1000.csv",
        output_file="data/Fan-out-reduction/Structure_prediction_TL_BRON_capec.xlsx",
        cve_col="CVE ID",
        relatedcwe_col="relatedcwe",
        cwe_capec_cwe_col="CWE-ID",
        cwe_capec_capec_col="Related_CAPEC_IDs",
        output_relatedcapec_col="relatedcapec",
        output_relatedcapec_count_col="relatedcapec_count",
    )
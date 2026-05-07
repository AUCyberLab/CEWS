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


def is_positive_label(x):
    """Return True if x represents a positive label (1, 1.0, true, yes)."""
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"1", "1.0", "true", "yes"}


def normalize_capec_id(x):
    """
    Normalize a single CAPEC value into the canonical 'CAPEC-N' form.
    Returns None if no number can be recovered.
    """
    if pd.isna(x):
        return None

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None

    if re.fullmatch(r"CAPEC-\d+", s, flags=re.IGNORECASE):
        num = re.search(r"\d+", s).group()
        return f"CAPEC-{num}"

    if re.fullmatch(r"\d+", s):
        return f"CAPEC-{s}"

    m = re.search(r"(\d+)", s)
    if m:
        return f"CAPEC-{m.group(1)}"

    return None


def extract_capec_ids(x):
    """
    Extract a list of canonical 'CAPEC-N' strings from a relatedcapec cell.

    Handles 'CAPEC-21, CAPEC-59', "['CAPEC-21', 'CAPEC-59']", '[21, 59]',
    list values, and NaN.
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
            # Fallback: pull every CAPEC-N or bare integer out of the raw string
            capec_nums = re.findall(r"CAPEC-(\d+)|\b(\d+)\b", s, flags=re.IGNORECASE)
            result = []
            seen = set()
            for pair in capec_nums:
                num = pair[0] if pair[0] else pair[1]
                cid = f"CAPEC-{num}"
                if cid not in seen:
                    seen.add(cid)
                    result.append(cid)
            return result

    result = []
    seen = set()
    for item in items:
        cid = normalize_capec_id(item)
        if cid and cid not in seen:
            seen.add(cid)
            result.append(cid)

    return result


def merge_unique_capecs(series):
    """Merge several CAPEC lists into one, preserving first-seen order."""
    seen = set()
    merged = []

    for lst in series:
        if not isinstance(lst, list):
            continue
        for item in lst:
            if item not in seen:
                seen.add(item)
                merged.append(item)

    return merged


def add_gtcapec_and_tp_fp(
    target_file,
    pair_file,
    output_file,
    cve_col="CVE-ID",
    pair_capec_col="CAPEC-ID",
    label_col="Label",
    relatedcapec_col="relatedcapec",
    output_gtcapec_col="GTcapec",
    output_tp_col="True_positive_count",
    output_fp_col="False_positive_count",
):
    """
    Build a ground-truth CAPEC list per CVE from `pair_file` (rows with
    Label=1), join it onto `target_file`, then count true / false positives
    relative to the predicted set in `relatedcapec`.
    """
    df_target = read_table(target_file)
    df_pair = read_table(pair_file)

    for col in [cve_col, relatedcapec_col]:
        if col not in df_target.columns:
            raise KeyError(f"Column not found in target_file: {col}")

    for col in [cve_col, pair_capec_col, label_col]:
        if col not in df_pair.columns:
            raise KeyError(f"Column not found in pair_file: {col}")

    df_target[cve_col] = df_target[cve_col].astype(str).str.strip()
    df_pair[cve_col] = df_pair[cve_col].astype(str).str.strip()

    # Keep only positive-labeled rows; these define the ground truth
    df_pos = df_pair[df_pair[label_col].apply(is_positive_label)].copy()

    df_pos["_norm_capec"] = df_pos[pair_capec_col].apply(normalize_capec_id)
    df_pos = df_pos.dropna(subset=["_norm_capec"]).copy()

    # Aggregate each CVE's GT CAPECs and join them back as a comma-joined string
    gt_df = (
        df_pos.groupby(cve_col)["_norm_capec"]
        .apply(lambda s: merge_unique_capecs([[x] for x in s.tolist()]))
        .reset_index()
    )

    gt_df[output_gtcapec_col] = gt_df["_norm_capec"].apply(
        lambda x: ", ".join(x) if x else ""
    )
    gt_df = gt_df.drop(columns=["_norm_capec"])

    df_out = df_target.merge(gt_df, on=cve_col, how="left")
    df_out[output_gtcapec_col] = df_out[output_gtcapec_col].fillna("")

    df_out["_gtcapec_list"] = df_out[output_gtcapec_col].apply(extract_capec_ids)
    df_out["_predcapec_list"] = df_out[relatedcapec_col].apply(extract_capec_ids)

    def calc_tp(gt_list, pred_list):
        gt_set = set(gt_list) if isinstance(gt_list, list) else set()
        pred_set = set(pred_list) if isinstance(pred_list, list) else set()
        return len(pred_set & gt_set)

    def calc_fp(gt_list, pred_list):
        gt_set = set(gt_list) if isinstance(gt_list, list) else set()
        pred_set = set(pred_list) if isinstance(pred_list, list) else set()
        return len(pred_set - gt_set)

    df_out[output_tp_col] = df_out.apply(
        lambda row: calc_tp(row["_gtcapec_list"], row["_predcapec_list"]),
        axis=1,
    )
    df_out[output_fp_col] = df_out.apply(
        lambda row: calc_fp(row["_gtcapec_list"], row["_predcapec_list"]),
        axis=1,
    )

    df_out = df_out.drop(columns=["_gtcapec_list", "_predcapec_list"], errors="ignore")

    write_table(df_out, output_file)

    print("========== Summary ==========")
    print(f"target_file total rows:           {len(df_target)}")
    print(f"pair_file positive rows:          {len(df_pos)}")
    print(f"Unique CVEs with matched GT:      {gt_df[cve_col].nunique()}")
    print()
    show_cols = [cve_col, relatedcapec_col, output_gtcapec_col, output_tp_col, output_fp_col]
    show_cols = [c for c in show_cols if c in df_out.columns]
    print(df_out[show_cols].head(10))

    return df_out


if __name__ == "__main__":
    add_gtcapec_and_tp_fp(
        target_file="data/Fan-out-reduction/Structure_prediction_TL_BRON_capec.xlsx",
        pair_file="data/Fan-out-reduction/Ensemble_Weighted_Results_Ranking.xlsx",
        output_file="data/Fan-out-reduction/BRON_tpfp.xlsx",
        cve_col="CVE-ID",
        pair_capec_col="CAPEC-ID",
        label_col="Label",
        relatedcapec_col="relatedcapec",
        output_gtcapec_col="GTcapec",
        output_tp_col="True_positive_count",
        output_fp_col="False_positive_count",
    )
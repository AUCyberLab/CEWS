import pandas as pd
import re
import os


# ================= Impact Extraction =================

def extract_impact(text):
    """
    Extract Impact information from a Consequences text field.

    1. Matches every "Impact: xxx" pattern (case-insensitive, line-bounded).
    2. When a single Impact contains comma-separated values (e.g. "Read Data, Gain Privileges"),
       splits them so each value is normalized individually.
    3. Deduplicates and returns a comma-joined string, or None if nothing was found.
    """
    if pd.isna(text) or str(text).strip() == "":
        return None

    matches = re.findall(r'Impact:\s*([^\n\r]+)', str(text), re.IGNORECASE)
    if not matches:
        return None

    cleaned_impacts = []
    for match in matches:
        parts = [p.strip() for p in match.split(',')]
        cleaned_impacts.extend(parts)

    unique_impacts = sorted(set(cleaned_impacts))
    return ", ".join(unique_impacts) if unique_impacts else None


def process_capec_impacts(input_file, output_file):
    """
    Stage 1: read the raw CAPEC database, extract the Impact column, and persist
    the trimmed table to disk. Other downstream scripts depend on this file, so
    we always write it out.

    Returns the resulting DataFrame so the next stage can consume it from memory.
    """
    print("=" * 50)
    print(f"Loading and processing: {input_file}")
    print("=" * 50)

    try:
        if input_file.endswith('.xlsx'):
            df = pd.read_excel(input_file)
        else:
            df = pd.read_csv(input_file, sep='\t')
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None

    required_cols = ['ID', 'Name', 'Consequences']
    for col in required_cols:
        if col not in df.columns:
            print(f"Missing required column '{col}'. Available columns: {list(df.columns)}")
            return None

    df['Impact'] = df['Consequences'].apply(extract_impact)
    new_df = df[['ID', 'Name', 'Impact']].copy()

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if output_file.endswith('.xlsx'):
        new_df.to_excel(output_file, index=False)
    else:
        new_df.to_csv(output_file, sep='\t', index=False)

    # ----- Summary -----
    has_impact_count = new_df['Impact'].notna().sum()
    no_impact_count = new_df['Impact'].isna().sum()

    all_unique_impacts = set()
    for impact_str in new_df['Impact'].dropna():
        parts = [p.strip() for p in impact_str.split(',')]
        all_unique_impacts.update(parts)
    all_unique_impacts.discard('')

    print(f"Done. Output saved to: {output_file}\n")
    print("--- Summary ---")
    print(f"Total CAPEC entries     : {len(new_df)}")
    print(f"Entries with Impact     : {has_impact_count}")
    print(f"Entries missing Impact  : {no_impact_count}")
    print(f"Distinct Impact types   : {len(all_unique_impacts)}\n")

    print("--- Impact Types ---")
    for i, impact_type in enumerate(sorted(all_unique_impacts), 1):
        print(f"{i}. {impact_type}")
    print("=" * 50)

    return new_df


# ================= Merge with CVE Data =================

def split_capec_ids(id_str):
    """Split a CAPEC-ID cell on commas or semicolons, trimming whitespace."""
    return [i.strip() for i in re.split(r'[;,]', id_str) if i.strip()]


def merge_with_cve(df_capec, cve_csv_path, mapping_xlsx_path, output_path):
    """
    Stage 2: join the cleaned CAPEC Impact table with the CVE descriptions and
    the (CVE-ID, CAPEC-ID) ground-truth mapping. CAPEC-ID cells that contain
    multiple IDs are exploded into separate rows before merging.
    """
    print("Loading CVE and mapping datasets...")
    try:
        df_cve = pd.read_csv(cve_csv_path, encoding='latin-1')
        df_mapping = pd.read_excel(mapping_xlsx_path)
    except FileNotFoundError as e:
        print(f"File not found, please check the path: {e}")
        return None

    print("Expanding multi-valued CAPEC-IDs...")
    df_mapping['CAPEC-ID'] = df_mapping['CAPEC-ID'].astype(str).fillna('')
    df_mapping['CAPEC-ID'] = df_mapping['CAPEC-ID'].apply(split_capec_ids)
    df_mapping = df_mapping.explode('CAPEC-ID')

    # Standardize column names
    df_capec = df_capec.rename(columns={
        'ID': 'CAPEC-ID',
        'Name': 'CAPEC_Name',
    })
    df_cve = df_cve.rename(columns={
        'CVE-Description': 'CVE_Description',
    })

    print("Merging datasets...")
    # Step 1: join the mapping table with the CVE data on 'CVE-ID'
    merged_df = pd.merge(
        df_mapping,
        df_cve[['CVE-ID', 'CVE_Description']],
        on='CVE-ID',
        how='left',
    )
    # Step 2: join the result with the CAPEC Impact data on 'CAPEC-ID'
    merged_df = pd.merge(
        merged_df,
        df_capec[['CAPEC-ID', 'CAPEC_Name', 'Impact']],
        on='CAPEC-ID',
        how='left',
    )

    final_columns = [
        'CVE-ID',
        'CVE_Description',
        'CAPEC-ID',
        'CAPEC_Name',
        'Impact',
    ]
    final_output_df = merged_df[final_columns]

    missing_cve = final_output_df['CVE_Description'].isna().sum()
    missing_capec = final_output_df['CAPEC_Name'].isna().sum()
    missing_impact = final_output_df['Impact'].isna().sum()

    print("Merge complete. Found:")
    print(f"   - {missing_cve} rows missing CVE description")
    print(f"   - {missing_capec} rows missing CAPEC name")
    print(f"   - {missing_impact} rows missing CAPEC Impact")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving result to: {output_path}")
    final_output_df.to_excel(output_path, index=False)
    print("Saved successfully.")

    return final_output_df


# ================= Pipeline =================

def build_impact_merged_dataset(
    raw_capec_path,
    capec_impacts_only_path,
    cve_csv_path,
    mapping_xlsx_path,
    final_output_path,
):
    """
    Full pipeline: extract Impact -> persist intermediate file (still consumed
    by other downstream scripts) -> merge with CVE data -> save final table.
    """
    df_capec = process_capec_impacts(raw_capec_path, capec_impacts_only_path)
    if df_capec is None:
        print("Stage 1 failed; aborting before merge.")
        return None

    return merge_with_cve(
        df_capec=df_capec,
        cve_csv_path=cve_csv_path,
        mapping_xlsx_path=mapping_xlsx_path,
        output_path=final_output_path,
    )


if __name__ == "__main__":
    RAW_CAPEC_FILE = "data/Impact/capec_database.xlsx"
    CAPEC_IMPACTS_ONLY_FILE = "data/Impact/CAPEC_Impacts_Only.xlsx"
    CVE_CSV_FILE = "data/Mechanism/CVE_CWE_withpageinfo_all.csv"
    MAPPING_FILE = "data/GT/656_cve_capec_mappings.xlsx"
    FINAL_OUTPUT_FILE = "data/Impact/Impact_merged_large.xlsx"

    build_impact_merged_dataset(
        raw_capec_path=RAW_CAPEC_FILE,
        capec_impacts_only_path=CAPEC_IMPACTS_ONLY_FILE,
        cve_csv_path=CVE_CSV_FILE,
        mapping_xlsx_path=MAPPING_FILE,
        final_output_path=FINAL_OUTPUT_FILE,
    )
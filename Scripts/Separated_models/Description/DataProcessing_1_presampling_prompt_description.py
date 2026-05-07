import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split


def process_and_sample(
    df,
    global_capec_dict,
    ood_capec_dict,
    output_excel,
    output_txt,
    num_negatives=1,
    split_name="Train",
    random_seed=42,
):
    
    random.seed(random_seed)
    cve_grouped = df.groupby('CVE-ID')
    ood_capec_ids = sorted(ood_capec_dict.keys())

    sampled_data = []

    print(f"[{split_name}] Generating samples for {len(cve_grouped)} unique CVEs...")

    for cve_id, group in cve_grouped:
        cve_desc = group['CVE_Description'].iloc[0]
        true_capecs = set(group['CAPEC-ID'].astype(str).str.strip().tolist())

        # Per-CVE global candidate pool: seen CAPECs minus this CVE's true labels
        global_candidates = sorted(set(global_capec_dict.keys()) - true_capecs)

        for _, row in group.iterrows():
            cid = str(row['CAPEC-ID']).strip()

            # --- Positive sample ---
            sampled_data.append({
                'CVE-ID': cve_id,
                'CVE_Description': cve_desc,
                'CAPEC-ID': cid,
                'CAPEC_Name': row['CAPEC_Name'],
                'CAPEC_Description': row['CAPEC_Description'],
                'Label': 1,
                'Expected_Score': "1.0",
                'Sample_Type': "POSITIVE",
                'Negative_Source': "",
            })

            # --- Global negatives ---
            if global_candidates:
                k = min(num_negatives, len(global_candidates))
                for neg_cid in random.sample(global_candidates, k):
                    neg_info = global_capec_dict[neg_cid]
                    sampled_data.append({
                        'CVE-ID': cve_id,
                        'CVE_Description': cve_desc,
                        'CAPEC-ID': neg_cid,
                        'CAPEC_Name': neg_info['CAPEC_Name'],
                        'CAPEC_Description': neg_info['CAPEC_Description'],
                        'Label': 0,
                        'Expected_Score': "0.0",
                        'Sample_Type': "NEGATIVE",
                        'Negative_Source': "GLOBAL",
                    })

            # --- OOD negatives ---
            if ood_capec_ids:
                k = min(num_negatives, len(ood_capec_ids))
                for neg_cid in random.sample(ood_capec_ids, k):
                    neg_info = ood_capec_dict[neg_cid]
                    sampled_data.append({
                        'CVE-ID': cve_id,
                        'CVE_Description': cve_desc,
                        'CAPEC-ID': neg_cid,
                        'CAPEC_Name': neg_info['CAPEC_Name'],
                        'CAPEC_Description': neg_info['CAPEC_Description'],
                        'Label': 0,
                        'Expected_Score': "0.0",
                        'Sample_Type': "NEGATIVE",
                        'Negative_Source': "OOD",
                    })

    final_df = pd.DataFrame(sampled_data)
    final_df = final_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Build TXT in the same order as the Excel rows
    txt_lines = []
    for index, row in final_df.iterrows():
        tag = row['Sample_Type']
        if row['Negative_Source']:
            tag = f"{tag}-{row['Negative_Source']}"
        txt_lines.append(
            f"========== [Row {index + 2} | {tag} | Expected: {row['Expected_Score']}] | {row['CVE-ID']} =========="
        )
        txt_lines.append(f"- CVE Description: {row['CVE_Description']}")
        txt_lines.append(f"- Target CAPEC Name: {row['CAPEC_Name']}")
        txt_lines.append(f"- Target CAPEC Description: {row['CAPEC_Description']}\n")

    # Drop helper columns from the Excel output
    excel_df = final_df.drop(columns=['Expected_Score', 'Sample_Type', 'Negative_Source'])

    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    excel_df.to_excel(output_excel, index=False)
    print(f"[{split_name}] Excel saved: {output_excel} ({len(excel_df)} rows)")

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_lines))
    print(f"[{split_name}] TXT saved:   {output_txt} (aligned with Excel order)\n")


def build_dataset_pipeline(
    input_excel,
    full_capec_excel,
    output_dir,
    output_tag,
    test_size=0.2,
    num_negatives=1,
    global_seed=42,
):
    """
    Pipeline: load positives -> build global pool (seen CAPECs) and OOD pool
    (CAPECs in the full dictionary but never seen as a positive) -> split by
    CVE-ID -> sample positives, global negatives, and OOD negatives together.
    """
    print(f"Loading positive samples: {input_excel}")
    df = pd.read_excel(input_excel)
    df = df.dropna(subset=['CVE-ID', 'CAPEC-ID'])

    # Global pool: every CAPEC that appears as a positive in the input data
    print("Building global CAPEC pool (seen CAPECs)...")
    global_capec_dict = {}
    for _, row in df.iterrows():
        cid = str(row['CAPEC-ID']).strip()
        if cid not in global_capec_dict and pd.notna(row['CAPEC_Name']):
            global_capec_dict[cid] = {
                'CAPEC_Name': row['CAPEC_Name'],
                'CAPEC_Description': row['CAPEC_Description'],
            }
    seen_capecs = set(global_capec_dict.keys())
    print(f"  -> {len(seen_capecs)} unique CAPECs in the global pool.")

    # OOD pool: CAPECs in the full dictionary that never appear as positives
    print(f"Loading full CAPEC dictionary: {full_capec_excel}")
    full_capec_df = pd.read_excel(full_capec_excel)
    has_description = 'Description' in full_capec_df.columns

    ood_capec_dict = {}
    for _, row in full_capec_df.iterrows():
        cid = str(row['ID']).strip()
        if cid in seen_capecs or pd.isna(row['Name']):
            continue
        description = row['Description'] if has_description else "None"
        ood_capec_dict[cid] = {
            'CAPEC_Name': row['Name'],
            'CAPEC_Description': description,
        }
    print(f"  -> {len(ood_capec_dict)} unseen CAPECs available for OOD negatives.\n")

    # Split by CVE-ID to prevent leakage between train and test
    unique_cves = df['CVE-ID'].unique()
    train_cves, test_cves = train_test_split(
        unique_cves, test_size=test_size, random_state=global_seed
    )
    print(f"Split: {len(train_cves)} train CVEs, {len(test_cves)} test CVEs\n")

    df_train = df[df['CVE-ID'].isin(train_cves)]
    df_test = df[df['CVE-ID'].isin(test_cves)]

    process_and_sample(
        df=df_train,
        global_capec_dict=global_capec_dict,
        ood_capec_dict=ood_capec_dict,
        output_excel=os.path.join(output_dir, f"Train_SFT_Samples_{output_tag}.xlsx"),
        output_txt=os.path.join(output_dir, f"Train_SFT_Prompt_{output_tag}.txt"),
        num_negatives=num_negatives,
        split_name="Train",
        random_seed=global_seed,
    )

    process_and_sample(
        df=df_test,
        global_capec_dict=global_capec_dict,
        ood_capec_dict=ood_capec_dict,
        output_excel=os.path.join(output_dir, f"Test_SFT_Samples_{output_tag}.xlsx"),
        output_txt=os.path.join(output_dir, f"Test_SFT_Prompt_{output_tag}.txt"),
        num_negatives=num_negatives,
        split_name="Test",
        random_seed=global_seed,
    )


# ================= Run =================
if __name__ == "__main__":
    INPUT_FILE = "data/Description/Description_merged_large.xlsx"
    FULL_CAPEC_DICTIONARY_FILE = "data/Description/capec_database.xlsx"
    OUTPUT_DIRECTORY = "data/Description/Processed_Data/"
    OUTPUT_TAG = "large_42"
    GLOBAL_SEED = 42

    build_dataset_pipeline(
        input_excel=INPUT_FILE,
        full_capec_excel=FULL_CAPEC_DICTIONARY_FILE,
        output_dir=OUTPUT_DIRECTORY,
        output_tag=OUTPUT_TAG,
        test_size=0.2,
        num_negatives=1,
        global_seed=GLOBAL_SEED,
    )
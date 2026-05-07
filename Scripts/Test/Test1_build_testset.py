import pandas as pd
import random
import os
import re

CAPEC_DB_PATH = "data/RankingEx/CAPEC_info/CAPEC_info_all.xlsx"     # Full official CAPEC database
CVE_DB_PATH = "data/Mechanism/CVE_CWE_withpageinfo_all.csv"         # CVE description database
UPDATED_TEST_GT_PATH = "data/Test_dataset/Test_set.xlsx"            # Updated Test Ground Truth mapping table

OUTPUT_EXCEL = "data/Test_dataset/Test_Set_with_neg.xlsx"
OUTPUT_TXT = "data/Test_dataset/New_Test_Prompt.txt"

GLOBAL_SEED = 42
NUM_NEGATIVES = 2 

def load_and_preprocess_data():
    print("Loading datasets...")
    # 1. Load CAPEC dictionary
    df_capec = pd.read_excel(CAPEC_DB_PATH)
    capec_rename_map = {
        'ID': 'CAPEC-ID',
        'Name': 'CAPEC_Name',
        'Description': 'CAPEC_Description',
        'Prerequisite': 'Prerequisite',
        'Mechanism': 'Mechanism',  
        'Impact': 'Impact'       
    }
    existing_cols = {k: v for k, v in capec_rename_map.items() if k in df_capec.columns}
    df_capec = df_capec.rename(columns=existing_cols)

    for col in ['CAPEC_Name', 'CAPEC_Description', 'CAPEC_Prerequisite', 'CAPEC_Mechanism', 'CAPEC_Impact']:
        if col not in df_capec.columns:
            df_capec[col] = 'None'
        else:
            df_capec[col] = df_capec[col].fillna('None')

    df_capec['CAPEC-ID'] = df_capec['CAPEC-ID'].astype(str).str.strip()
    capec_dict = df_capec.set_index('CAPEC-ID').to_dict('index')

    # 2. Load CVE descriptions
    df_cve = pd.read_csv(CVE_DB_PATH, encoding='latin-1', low_memory=False)
    df_cve = df_cve.rename(columns={'CVE-Description': 'CVE_Description'})
    df_cve = df_cve[['CVE-ID', 'CVE_Description']].dropna().drop_duplicates(subset=['CVE-ID'])

    # 3. Load and explode the Ground Truth
    df_gt = pd.read_excel(UPDATED_TEST_GT_PATH)
    df_gt['CAPEC-ID'] = df_gt['CAPEC-ID'].astype(str).fillna('')
    df_gt['CAPEC-ID'] = df_gt['CAPEC-ID'].apply(lambda x: [i.strip() for i in re.split(r'[;,]', x) if i.strip()])
    df_gt = df_gt.explode('CAPEC-ID')

    return df_gt, df_cve, capec_dict

def generate_test_set(df_gt, df_cve, capec_dict):
    random.seed(GLOBAL_SEED)
    all_capec_ids = sorted(list(capec_dict.keys()))

    if 'CVE_Description' in df_gt.columns:
        df_gt = df_gt.drop(columns=['CVE_Description'])

    merged_positives = pd.merge(df_gt, df_cve, on='CVE-ID', how='inner')
    cve_grouped = merged_positives.groupby('CVE-ID')
    print(merged_positives.columns)
    sampled_data = []
    print(f"Generating test set samples... ({len(cve_grouped)} unique CVEs in total)")

    for cve_id, group in cve_grouped:
        cve_desc = group['CVE_Description'].iloc[0]
        true_capecs = set(group['CAPEC-ID'].astype(str).str.strip().tolist())

        # --- Record positive samples ---
        for cid in true_capecs:
            if cid in capec_dict:
                info = capec_dict[cid]
                sampled_data.append({
                    'CVE-ID': cve_id,
                    'CVE_Description': cve_desc,
                    'CAPEC-ID': cid,
                    'CAPEC_Name': info['CAPEC_Name'],
                    'CAPEC_Description': info['CAPEC_Description'],
                    'CAPEC_Prerequisite': info['CAPEC_Prerequisite'],
                    'CAPEC_Mechanism': info['CAPEC_Mechanism'],
                    'CAPEC_Impact': info['CAPEC_Impact'],
                    'Label': 1,
                    'Expected_Score': "1.0",
                    'Sample_Type': "POSITIVE"
                })

        # --- Generate negative samples ---
        candidate_negatives = list(set(all_capec_ids) - true_capecs)
        candidate_negatives.sort()

        if candidate_negatives:
            sample_size = min(NUM_NEGATIVES * len(true_capecs), len(candidate_negatives))
            sampled_neg_ids = random.sample(candidate_negatives, sample_size)

            for neg_cid in sampled_neg_ids:
                neg_info = capec_dict[neg_cid]
                sampled_data.append({
                    'CVE-ID': cve_id,
                    'CVE_Description': cve_desc,
                    'CAPEC-ID': neg_cid,
                    'CAPEC_Name': neg_info['CAPEC_Name'],
                    'CAPEC_Description': neg_info['CAPEC_Description'],
                    'CAPEC_Prerequisite': neg_info['CAPEC_Prerequisite'],
                    'CAPEC_Mechanism': neg_info['CAPEC_Mechanism'],
                    'CAPEC_Impact': neg_info['CAPEC_Impact'],
                    'Label': 0,
                    'Expected_Score': "0.0",
                    'Sample_Type': "NEGATIVE"
                })

    final_df = pd.DataFrame(sampled_data)
    final_df = final_df.sample(frac=1, random_state=GLOBAL_SEED).reset_index(drop=True)

    return final_df

def export_results(final_df, output_excel, output_txt):
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)

    txt_lines = []
    for index, row in final_df.iterrows():
        txt_lines.append(f"========== [Row {index + 2} | {row['Sample_Type']} | Expected: {row['Expected_Score']}] | {row['CVE-ID']} ==========")
        txt_lines.append(f"- CVE Description: {row['CVE_Description']}")
        txt_lines.append(f"- Target CAPEC ID: {row['CAPEC-ID']}")
        txt_lines.append(f"- Target CAPEC Name: {row['CAPEC_Name']}")
        txt_lines.append(f"- Target CAPEC Description: {row['CAPEC_Description']}")
        txt_lines.append(f"- Target CAPEC Prerequisite: {row['CAPEC_Prerequisite']}")
        txt_lines.append(f"- Target CAPEC Mechanism: {row['CAPEC_Mechanism']}")
        txt_lines.append(f"- Target CAPEC Impact: {row['CAPEC_Impact']}\n")

    excel_df = final_df.drop(columns=['Expected_Score', 'Sample_Type'])
    excel_df.to_excel(output_excel, index=False)
    print(f"Excel test set exported successfully: {output_excel} ({len(excel_df)} samples in total)")

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_lines))
    print(f"TXT reference file exported successfully: {output_txt} (order fully aligned with Excel)\n")

if __name__ == "__main__":
    df_gt, df_cve, capec_dict = load_and_preprocess_data()
    final_test_df = generate_test_set(df_gt, df_cve, capec_dict)
    export_results(final_test_df, OUTPUT_EXCEL, OUTPUT_TXT)
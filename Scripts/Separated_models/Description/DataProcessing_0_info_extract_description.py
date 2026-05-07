import pandas as pd
import os
import re

# ================= 1. File Paths =================
DATASET1_PATH = "data/Description/capec_database.xlsx"
DATASET2_PATH = "data/Mechanism/CVE_CWE_withpageinfo_all.csv"
DATASET3_PATH = "data/GT/656_cve_capec_mappings.xlsx"
OUTPUT_PATH = "data/Description/Description_merged_large.xlsx"

# ================= 2. Load Datasets =================
print("Loading datasets...")
try:
    df_capec = pd.read_excel(DATASET1_PATH)
    df_cve = pd.read_csv(DATASET2_PATH, encoding='latin-1')
    df_mapping = pd.read_excel(DATASET3_PATH)
except FileNotFoundError as e:
    print(f"File not found, please check the path: {e}")
    exit(1)

# ================= 3. Preprocess and Rename =================
print("Expanding multi-valued CAPEC-IDs...")

df_mapping['CAPEC-ID'] = df_mapping['CAPEC-ID'].astype(str).fillna('')

def split_capec_ids(id_str):
    return [i.strip() for i in re.split(r'[;,]', id_str) if i.strip()]

df_mapping['CAPEC-ID'] = df_mapping['CAPEC-ID'].apply(split_capec_ids)
df_mapping = df_mapping.explode('CAPEC-ID')

df_capec = df_capec.rename(columns={
    'ID': 'CAPEC-ID',
    'Name': 'CAPEC_Name',
    'Description': 'CAPEC_Description'
})

df_cve = df_cve.rename(columns={
    'CVE-Description': 'CVE_Description'
})

# ================= 4. Merge (Left Join) =================
print("Merging datasets...")

merged_df = pd.merge(
    df_mapping,
    df_cve[['CVE-ID', 'CVE_Description']],
    on='CVE-ID',
    how='left'
)

merged_df = pd.merge(
    merged_df,
    df_capec[['CAPEC-ID', 'CAPEC_Name', 'CAPEC_Description']],
    on='CAPEC-ID',
    how='left'
)

# ================= 5. Finalize Columns and Save =================
final_columns = [
    'CVE-ID',
    'CVE_Description',
    'CAPEC-ID',
    'CAPEC_Name',
    'CAPEC_Description'
]

final_output_df = merged_df[final_columns]

missing_cve = final_output_df['CVE_Description'].isna().sum()
missing_capec = final_output_df['CAPEC_Name'].isna().sum()
print(f"Merge complete. Found {missing_cve} rows missing CVE description, {missing_capec} rows missing CAPEC details.")

output_dir = os.path.dirname(OUTPUT_PATH)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Saving result to: {OUTPUT_PATH}")
final_output_df.to_excel(OUTPUT_PATH, index=False)
print("Saved successfully!")
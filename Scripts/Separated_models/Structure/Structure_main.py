import pandas as pd
import json
import ast 
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from pipeline import VulnerabilityIntelligencePipeline
from ScoringFunctions import CVECAPECScoringSystem


CONFIG = {
    # Model and feature mapping
    "model_weight_path": "./V2W-BERT/Results/Model/V2WBERT-LINK-distilbert-base-uncased-auto_BEST",
    "pretrained_model_name": "distilbert-base-uncased",
    "label2idx_path": "./V2W-BERT/Dataset/CWE/label2idx_cwe.json",

    # Structured knowledge graph data
    "cwe_parsed_data": "data/Structure/cwe_parsed_data.xlsx",
    "capec_parsed_data": "data/Structure/capec_parsed_data.xlsx",
    "v2w_dataset_dir": "./V2W-BERT/Dataset/CWE/processed/",

    # Evaluation task input/output paths
    "known_cwes_file": "data/Mechanism/CVE_CWE_withpageinfo_all.csv",
    "input_pairs_file": "data/Test_dataset/Test_hard.xlsx",
    "output_excel_path": "data/Test_dataset/output/Hard/Structure_prediction_hard.xlsx",
}


def build_static_knowledge_graph(cwe_excel_path, capec_excel_path):
    """Build the static base knowledge graph from CWE and CAPEC structure data."""
    scorer = CVECAPECScoringSystem()
    print("Initializing graph base, importing structure data...")

    # 1. Import CWE structure and CWE-CAPEC mappings
    try:
        cwe_df = pd.read_excel(cwe_excel_path)
        for _, row in cwe_df.iterrows():
            cwe_id = str(row['CWE ID']).strip()
            if cwe_id == 'nan' or not cwe_id:
                continue

            cwe_rels = str(row['Relationships (CWE)'])
            if cwe_rels != 'nan' and cwe_rels.strip():
                for rel_line in cwe_rels.split('\n'):
                    parts = rel_line.strip().split(':')
                    if len(parts) == 2:
                        rel_type = parts[0].strip()
                        target_cwe = parts[1].strip()
                        if rel_type == 'ChildOf':
                            scorer.add_cwe_hierarchy(parent_cwe=target_cwe, child_cwe=cwe_id)

            related_capecs = str(row['Related CAPECs'])
            if related_capecs != 'nan' and related_capecs.strip():
                for capec in related_capecs.split(','):
                    capec_id = capec.strip()
                    if capec_id:
                        scorer.add_cwe_capec_edge(cwe_id=cwe_id, capec_id=capec_id)

        print("CWE structure and mappings imported.")
    except Exception as e:
        print(f"Error importing CWE data: {e}")

    # 2. Import CAPEC internal relationships and supplementary mappings
    try:
        capec_df = pd.read_excel(capec_excel_path)
        for _, row in capec_df.iterrows():
            capec_id = str(row['CAPEC ID']).strip()
            if capec_id == 'nan' or not capec_id:
                continue

            capec_rels = str(row['Relationships'])
            if capec_rels != 'nan' and capec_rels.strip():
                for rel_line in capec_rels.split('\n'):
                    parts = rel_line.strip().split(':')
                    if len(parts) == 2:
                        rel_type = parts[0].strip()
                        target_capec = parts[1].strip()
                        scorer.add_capec_relationship(
                            capec_1=capec_id,
                            capec_2=target_capec,
                            rel_type=rel_type,
                        )

            related_cwes = str(row['Related CWEs'])
            if related_cwes != 'nan' and related_cwes.strip():
                for cwe in related_cwes.split(','):
                    cwe_target_id = cwe.strip()
                    if cwe_target_id:
                        scorer.add_cwe_capec_edge(cwe_id=cwe_target_id, capec_id=capec_id)

        print("CAPEC relationships and supplementary mappings imported.")
    except Exception as e:
        print(f"Error importing CAPEC data: {e}")

    return scorer


def main(mode="generate_pool"):
    """
    Modes:
      - "evaluate_pairs": run evaluation on specific CVE-CAPEC pairs (experiment type 1)
      - "generate_pool" : filter and generate Top-K CAPEC candidate pools per CVE (experiment type 2)
    """
    print(f"=== Vulnerability analysis system started | Mode: {mode} ===")

    build_graph_fn = lambda: build_static_knowledge_graph(
        cwe_excel_path=CONFIG['cwe_parsed_data'],
        capec_excel_path=CONFIG['capec_parsed_data'],
    )

    pipeline = VulnerabilityIntelligencePipeline(
        model_weight_path=CONFIG['model_weight_path'],
        build_static_graph_fn=build_graph_fn,
    )

    cve_to_cwes_map = {}
    try:
        file_path = CONFIG['known_cwes_file']
        print(f"Loading known CWE mappings from: {file_path}")

        if file_path.endswith('.csv'):
            try:
                df_cwes = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                print("Special characters detected, retrying with error-tolerant encoding...")
                df_cwes = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')
        else:
            df_cwes = pd.read_excel(file_path)

        for _, row in df_cwes.iterrows():
            cve_id = str(row.get('CVE-ID')).strip()
            if cve_id != 'nan' and cve_id:
                raw_cwes = row.get('CWE-ID')
                cleaned_cwes = []

                if pd.notna(raw_cwes):
                    raw_str = str(raw_cwes).strip()
                    try:
                        if raw_str.startswith('['):
                            parsed_list = ast.literal_eval(raw_str)
                            if isinstance(parsed_list, list):
                                cleaned_cwes = [
                                    str(c).strip()
                                    for c in parsed_list
                                    if str(c).strip().upper() != 'UNKNOWN'
                                ]
                        else:
                            cleaned_cwes = [
                                c.strip()
                                for c in raw_str.split(',')
                                if c.strip() and c.strip().upper() != 'UNKNOWN'
                            ]
                    except (ValueError, SyntaxError):
                        clean_str = raw_str.replace('[', '').replace(']', '').replace("'", "").replace('"', '')
                        cleaned_cwes = [
                            c.strip()
                            for c in clean_str.split(',')
                            if c.strip() and c.strip().upper() != 'UNKNOWN'
                        ]

                cve_to_cwes_map[cve_id] = cleaned_cwes

        print(f"Loaded valid CWEs for {len(cve_to_cwes_map)} CVEs.")
    except Exception as e:
        print(f"Error reading CWE mapping file: {e}")
        return

    # ==========================================
    # Load CVE evaluation input
    # ==========================================
    try:
        print(f"Loading evaluation tasks from: {CONFIG['input_pairs_file']}")
        if CONFIG['input_pairs_file'].endswith('.csv'):
            try:
                df_input = pd.read_csv(CONFIG['input_pairs_file'], encoding='utf-8')
            except UnicodeDecodeError:
                df_input = pd.read_csv(CONFIG['input_pairs_file'], encoding='utf-8', encoding_errors='replace')
        else:
            df_input = pd.read_excel(CONFIG['input_pairs_file'])
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    all_results = []

    # ==========================================
    # Run pipeline based on mode
    # ==========================================
    if mode == "evaluate_pairs":
        total_pairs = len(df_input)
        print(f"Loaded {total_pairs} CVE-CAPEC pairs, starting evaluation...\n")

        for index, row in df_input.iterrows():
            cve_id = str(row.get('CVE-ID')).strip()
            target_capec = str(row.get('CAPEC-ID')).strip()

            if cve_id == 'nan' or not cve_id or target_capec == 'nan' or not target_capec:
                continue

            description = str(row.get('CVE_Description')).strip()
            if description == 'nan':
                description = ""

            known_cwes = cve_to_cwes_map.get(cve_id, [])

            if (index + 1) % 50 == 0:
                print(f"Processed {index + 1} / {total_pairs}...")

            result = pipeline.evaluate_specific_pair(
                cve_id=cve_id,
                target_capec=target_capec,
                description=description,
                known_cwes=known_cwes,
            )
            all_results.append(result)

        df_results = pd.DataFrame(all_results)
        df_results.to_excel(CONFIG['output_excel_path'], index=False)
        print(f"\nEvaluation complete. Results saved to: {CONFIG['output_excel_path']}")

    elif mode == "generate_pool":
        # Candidate pool generation only needs each unique CVE once
        df_unique_cves = df_input.drop_duplicates(subset=['CVE-ID']).copy()
        total_cves = len(df_unique_cves)
        print(f"After dedup: {total_cves} unique CVEs. Starting Top-K CAPEC filtering...\n")

        for index, row in df_unique_cves.iterrows():
            cve_id = str(row.get('CVE-ID')).strip()
            if cve_id == 'nan' or not cve_id:
                continue

            description = str(row.get('CVE_Description')).strip()
            if description == 'nan':
                description = ""

            known_cwes = cve_to_cwes_map.get(cve_id, [])

            pool_result = pipeline.generate_candidate_pool(
                cve_id=cve_id,
                description=description,
                known_cwes=known_cwes,
                top_k=30,
                enable_fallback=True,
            )

            # Flatten the candidate list into rows so it's easy to inspect or re-rank in Excel
            for candidate in pool_result['Candidate Pool']:
                all_results.append({
                    "CVE-ID": cve_id,
                    "Link Status": pool_result['Strategy Triggered'],
                    "Candidate CAPEC": candidate['CAPEC ID'],
                    "Structure Rank": candidate['Rank'],
                    "Structure Score": candidate['Score'],
                })

        df_results = pd.DataFrame(all_results)
        output_pool_path = "data/Structure/Candidate_Pool_Top30.xlsx"
        df_results.to_excel(output_pool_path, index=False)
        print(f"\nCandidate pool generation complete. {len(all_results)} candidate rows saved to: {output_pool_path}")


if __name__ == "__main__":
    main(mode="evaluate_pairs")
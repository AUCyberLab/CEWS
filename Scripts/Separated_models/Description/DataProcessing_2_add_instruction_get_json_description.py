import pandas as pd
import json
import os


def convert_table_to_finetuning_json(input_file_path, output_json_path):
    """
    Read a table containing CVE and CAPEC data and convert it into the JSON
    format required for LLM fine-tuning.
    """
    print(f"Loading file: {input_file_path}")

    df = pd.read_excel(input_file_path)
    df.fillna('', inplace=True)

    system_instruction = (
        "You are an expert cybersecurity analyst. Your task is to evaluate the logical and semantic match between a `CVE Description` and a `Target CAPEC Description`. Your goal is to determine if the vulnerability described in the CVE represents a valid instance of the overarching attack pattern described in the CAPEC.\n\n"
        "Determine a match score: 1.0 (Strongly/Explicitly Matched), 0.5 (Implicitly/Partially Matched), or 0.0 (Mismatched/Contradicted).\n\n"
        "You must first provide a step-by-step reasoning process enclosed strictly within <think></think> tags, covering: [Step 1: CAPEC Profile Extraction], [Step 2: CVE Vulnerability Mapping], [Step 3: Alignment Analysis], and [Step 4: Scoring Decision]. After the reasoning block, output exactly one JSON object containing the \"match_score\"."
    )

    formatted_data = []

    output_col_name = df.columns[-1]

    for _, row in df.iterrows():
        description_text = row.get('CAPEC_Description', '')

        input_text = (
            f"Target CAPEC:\n"
            f"CAPEC-ID: {row.get('CAPEC-ID', '')}\n"
            f"CAPEC Name: {row.get('CAPEC_Name', '')}\n"
            f"CAPEC Description: {description_text}\n\n"
            f"Evidence to Evaluate:\n"
            f"CVE-ID: {row.get('CVE-ID', '')}\n"
            f"CVE Description: {row.get('CVE_Description', '')}"
        )

        raw_output = str(row[output_col_name]).replace('""', '"')

        formatted_data.append({
            "instruction": system_instruction,
            "input": input_text,
            "output": raw_output,
        })

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(formatted_data)} rows. Saved to {output_json_path}")


if __name__ == "__main__":
    input_file = 'data/Description/Description_train.xlsx'
    output_file = 'data/Description/Description_train.json'

    convert_table_to_finetuning_json(input_file, output_file)
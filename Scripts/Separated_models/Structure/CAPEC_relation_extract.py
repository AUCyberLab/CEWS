import xml.etree.ElementTree as ET
import pandas as pd


def extract_capec_to_excel(xml_file_path, output_excel_path):
    print(f"Parsing file: {xml_file_path}")

    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Failed to parse XML: {e}")
        return

    ns = {'capec': 'http://capec.mitre.org/capec-3'}
    data = []

    for ap in root.findall('.//capec:Attack_Pattern', ns):
        raw_id = ap.get('ID')
        capec_id = f"CAPEC-{raw_id}" if raw_id else ""
        status = ap.get('Status', '')

        desc_elem = ap.find('capec:Description', ns)
        description = ""
        if desc_elem is not None:
            text_fragments = [text.strip() for text in desc_elem.itertext() if text.strip()]
            description = "\n".join(text_fragments)

        # Skip deprecated entries
        if description.lower().startswith("deprecated") or status.lower() == "deprecated":
            continue

        # Related CAPECs
        related_capecs = []
        relationships = []
        rap_container = ap.find('capec:Related_Attack_Patterns', ns)
        if rap_container is not None:
            for rap in rap_container.findall('capec:Related_Attack_Pattern', ns):
                rel_nature = rap.get('Nature')
                raw_rel_id = rap.get('CAPEC_ID')
                if raw_rel_id:
                    formatted_rel_id = f"CAPEC-{raw_rel_id}"
                    related_capecs.append(formatted_rel_id)
                    if rel_nature:
                        relationships.append(f"{rel_nature}: {formatted_rel_id}")

        # Related CWEs
        related_cwes = []
        rw_container = ap.find('capec:Related_Weaknesses', ns)
        if rw_container is not None:
            for rw in rw_container.findall('capec:Related_Weakness', ns):
                raw_cwe_id = rw.get('CWE_ID')
                if raw_cwe_id:
                    related_cwes.append(f"CWE-{raw_cwe_id}")

        data.append({
            'CAPEC ID': capec_id,
            'Description': description,
            'Related CAPECs': ", ".join(related_capecs),
            'Relationships': "\n".join(relationships),
            'Related CWEs': ", ".join(related_cwes),
        })

    if not data:
        print("No valid data extracted.")
        return

    df = pd.DataFrame(data)

    try:
        df.to_excel(output_excel_path, index=False)
        print(f"Extracted {len(df)} CAPEC entries. Saved to: {output_excel_path}")
    except Exception as e:
        print(f"Failed to save Excel: {e}")


if __name__ == "__main__":
    input_xml = "data/Structure/capec_latest.xml"
    output_xlsx = "data/Structure/capec_parsed_data.xlsx"

    extract_capec_to_excel(input_xml, output_xlsx)
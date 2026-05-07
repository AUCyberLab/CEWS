import os
import re
import io
import zipfile
import requests
import xml.etree.ElementTree as ET
import pandas as pd


def download_latest_cwe_xml(download_dir="data/Structure"):
    """
    Download the latest CWE Comprehensive Dictionary (XML, distributed as a ZIP)
    from the official MITRE site, extract it into `download_dir`, and return
    the path to the extracted XML file.
    """
    url = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
    print(f"Downloading the latest CWE archive...\nURL: {url}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
            xml_files = [name for name in thezip.namelist() if name.endswith('.xml')]
            if not xml_files:
                raise ValueError("No XML file found inside the downloaded archive.")

            xml_filename = xml_files[0]
            os.makedirs(download_dir, exist_ok=True)
            xml_filepath = os.path.join(download_dir, xml_filename)

            with open(xml_filepath, 'wb') as f:
                f.write(thezip.read(xml_filename))

        print(f"Download and extraction complete: {xml_filepath}")
        return xml_filepath

    except requests.exceptions.RequestException as e:
        print(f"Network request failed: {e}")
        return None
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None


def extract_cwe_to_excel(xml_file_path, output_excel_path):
    """
    Parse a CWE XML file, extract weakness entries, and save them to an Excel
    file. Handles the dynamic XML namespace (so it survives MITRE version
    bumps) and filters out deprecated entries.
    """
    print(f"Parsing file: {xml_file_path}")

    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Failed to parse XML: {e}")
        return

    # Read the namespace dynamically from the root tag so the parser
    # keeps working when MITRE bumps the schema version (e.g. cwe-6 -> cwe-7).
    ns_match = re.match(r'\{.*\}', root.tag)
    ns_uri = ns_match.group(0).strip('{}') if ns_match else 'http://cwe.mitre.org/cwe-6'
    ns = {'cwe': ns_uri}

    data = []

    for weakness in root.findall('.//cwe:Weaknesses/cwe:Weakness', ns):
        raw_id = weakness.get('ID')
        cwe_id = f"CWE-{raw_id}" if raw_id else ""
        status = weakness.get('Status', '')

        # Description: recursively join every non-empty text fragment
        desc_elem = weakness.find('cwe:Description', ns)
        description = ""
        if desc_elem is not None:
            text_fragments = [text.strip() for text in desc_elem.itertext() if text.strip()]
            description = "\n".join(text_fragments)

        # Skip deprecated entries
        if description.lower().startswith("deprecated") or status.lower() == "deprecated":
            continue

        # Related Weaknesses: parent/child/peer relationships
        related_cwes = []
        rw_container = weakness.find('cwe:Related_Weaknesses', ns)
        if rw_container is not None:
            for rw in rw_container.findall('cwe:Related_Weakness', ns):
                rel_nature = rw.get('Nature')  # ChildOf, ParentOf, PeerOf, ...
                raw_rel_id = rw.get('CWE_ID')
                if raw_rel_id:
                    formatted_rel_id = f"CWE-{raw_rel_id}"
                    if rel_nature:
                        related_cwes.append(f"{rel_nature}: {formatted_rel_id}")
                    else:
                        related_cwes.append(formatted_rel_id)

        # Related Attack Patterns: linked CAPECs
        related_capecs = []
        rap_container = weakness.find('cwe:Related_Attack_Patterns', ns)
        if rap_container is not None:
            for rap in rap_container.findall('cwe:Related_Attack_Pattern', ns):
                raw_capec_id = rap.get('CAPEC_ID')
                if raw_capec_id:
                    related_capecs.append(f"CAPEC-{raw_capec_id}")

        data.append({
            'CWE ID': cwe_id,
            'Name': weakness.get('Name', ''),
            'Description': description,
            'Relationships (CWE)': "\n".join(related_cwes),
            'Related CAPECs': ", ".join(related_capecs),
        })

    if not data:
        print("No valid entries extracted.")
        return

    df = pd.DataFrame(data)

    output_dir = os.path.dirname(output_excel_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        df.to_excel(output_excel_path, index=False)
        print(f"Extracted {len(df)} CWE entries. Saved to: {output_excel_path}")
    except Exception as e:
        print(f"Failed to save Excel: {e}")


if __name__ == "__main__":
    output_xlsx = "data/Structure/cwe_parsed_data.xlsx"

    downloaded_xml_path = download_latest_cwe_xml()

    if downloaded_xml_path:
        extract_cwe_to_excel(downloaded_xml_path, output_xlsx)
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any


def merge_and_clean_excel_files(
    file_paths: List[str],
    output_path: Optional[str] = None,
    sheet_name: Any = 0,
    keep_source_info: bool = True,
    replacement_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Concatenate multiple Excel files in the given order and clean special
    characters from any string cells.

    Default replacement rules:
    - Ċ -> newline
    - Ġ -> space

    Args:
        file_paths:       List of Excel file paths to merge.
        output_path:      If provided, the merged result is also written here.
        sheet_name:       Which sheet to read; defaults to the first one.
        keep_source_info: If True, prepend source_file and source_order columns.
        replacement_map:  Custom char-substitution map; falls back to defaults if None.

    Returns:
        The merged and cleaned DataFrame.
    """
    if len(file_paths) != 10:
        raise ValueError(f"Expected 10 file paths, got {len(file_paths)}.")

    if replacement_map is None:
        replacement_map = {
            "Ċ": "\n",
            "Ġ": " ",
        }

    all_dfs = []

    for idx, file_path in enumerate(file_paths, start=1):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # Apply replacements only to string cells, leaving other dtypes untouched
        df = df.map(
            lambda x: _clean_text(x, replacement_map) if isinstance(x, str) else x
        )

        if keep_source_info:
            df.insert(0, "source_file", path.name)
            df.insert(1, "source_order", idx)

        all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)

    if output_path:
        merged_df.to_excel(output_path, index=False)

    return merged_df


def _clean_text(text: str, replacement_map: Dict[str, str]) -> str:
    """Apply every (old -> new) substitution from replacement_map to text."""
    for old, new in replacement_map.items():
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    files = [
        "data/RankingEx/output/Mechanism_predictions_TL_01.xlsx",
        "data/RankingEx/output/Mechanism_predictions_TL_02.xlsx",
        "data/RankingEx/output/Mechanism_predictions_TL_03.xlsx",
        "data/RankingEx/output/Mechanism_predictions_TL_04.xlsx",
        "data/RankingEx/output/Mechanism_predictions_TL_05.xlsx",
        "data/RankingEx/output/Mechanism_predictions_TL_06.xlsx",
        "data/RankingEx/output/Mechanism_predictions_TL_07.xlsx",
        "data/RankingEx/output/Mechanism_predictions_TL_08.xlsx",
        "data/RankingEx/output/Mechanism_predictions_TL_09.xlsx",
        "data/RankingEx/output/Mechanism_predictions_TL_10.xlsx",
    ]

    merged_df = merge_and_clean_excel_files(
        file_paths=files,
        output_path="data/RankingEx/output/Mechanism_predictions_TL_merged.xlsx",
    )

    print(merged_df.head())
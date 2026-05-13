import re
import pandas as pd


def extract_match_score_column(
    df: pd.DataFrame,
    raw_col: str = "Raw_Response",
    score_col: str = "Match_Score",
) -> pd.DataFrame:
    """
    Extract the numeric match_score from each row's raw response and write it
    into a dedicated column.

    Recognizes patterns like:
        "match_score": 0.0
        'match_score': 1
        "match_score": 0.5

    Args:
        df:        Input DataFrame.
        raw_col:   Column holding the raw model response.
        score_col: Column to write the extracted score into.

    Returns:
        The same DataFrame with the score column added/overwritten.
    """
    pattern = re.compile(
        r'["\']match_score["\']\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        re.IGNORECASE,
    )

    def extract_score(text):
        if pd.isna(text):
            return None
        match = pattern.search(str(text))
        if match:
            return float(match.group(1))
        return None

    df[score_col] = df[raw_col].map(extract_score)
    return df


if __name__ == "__main__":
    input_path = "data/RankingEx/output/Prerequisite_predictions_TL_merged.xlsx"
    output_path = input_path  # Overwrite in place

    df = pd.read_excel(input_path)
    df = extract_match_score_column(df)
    df.to_excel(output_path, index=False)
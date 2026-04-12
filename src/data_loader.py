"""
Data loading module — reads the IMDB CSV from the dataset folder.
"""
import os
import pandas as pd


def find_csv_path() -> str:
    """Locate the IMDB CSV file in the project directory tree."""
    candidates = [
        os.path.join("data", "imdb_top_1000.csv"),
        os.path.join("imdb_dataset", "imdb_top_1000.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find imdb_top_1000.csv. "
        "Place it in data/ or imdb_dataset/ directory."
    )


def load_raw_dataframe(path: str = None) -> pd.DataFrame:
    """Load the raw CSV into a pandas DataFrame."""
    if path is None:
        path = find_csv_path()
    df = pd.read_csv(path, encoding="latin-1")
    return df

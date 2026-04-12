"""
Data preprocessing — cleans and normalizes the IMDB dataset.

Handles: Released_Year, Gross, Runtime, Genre, Cast, Meta_score, No_of_Votes.
Produces a clean DataFrame and text-ready representations for embeddings.
"""
import pandas as pd
import numpy as np
import re


def clean_released_year(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Released_Year: strip non-numeric entries, cast to numeric."""
    df = df.copy()
    df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce")
    df["Released_Year"] = df["Released_Year"].astype("Int64")  # nullable int
    return df


def clean_runtime(df: pd.DataFrame) -> pd.DataFrame:
    """Strip ' min' suffix from Runtime and cast to int."""
    df = df.copy()
    df["Runtime"] = (
        df["Runtime"]
        .astype(str)
        .str.replace(" min", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )
    return df


def clean_gross(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Gross column:
    - Strip commas and quotes, cast to float.
    - Preserve NaN for missing values (do NOT fill with zero).
    - Add a has_gross boolean column.
    """
    df = df.copy()

    def _parse_gross(val):
        if pd.isna(val) or str(val).strip() == "":
            return np.nan
        cleaned = str(val).replace(",", "").replace('"', "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return np.nan

    df["Gross"] = df["Gross"].apply(_parse_gross)
    df["has_gross"] = df["Gross"].notna()
    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert IMDB_Rating, Meta_score, No_of_Votes to numeric."""
    df = df.copy()
    df["IMDB_Rating"] = pd.to_numeric(df["IMDB_Rating"], errors="coerce")
    df["Meta_score"] = pd.to_numeric(df["Meta_score"], errors="coerce")
    df["No_of_Votes"] = (
        df["No_of_Votes"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )
    return df


def clean_genre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split Genre into a list column (genre_list).
    Keep original Genre column for display.
    """
    df = df.copy()
    df["genre_list"] = (
        df["Genre"]
        .fillna("")
        .apply(lambda g: [x.strip() for x in g.split(",") if x.strip()])
    )
    return df


def clean_cast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine Star1-Star4 into cast_list. Add lead_actor alias for Star1.
    Strip whitespace from all names.
    """
    df = df.copy()
    star_cols = ["Star1", "Star2", "Star3", "Star4"]
    for col in star_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["cast_list"] = df[star_cols].apply(
        lambda row: [name for name in row if name and name != "nan"], axis=1
    )
    df["lead_actor"] = df["Star1"]
    return df


def clean_director(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from Director column."""
    df = df.copy()
    df["Director"] = df["Director"].astype(str).str.strip()
    return df


def build_text_representation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a text-ready representation for semantic indexing.
    Format: "{Series_Title} ({Released_Year}) — {Genre} — Directed by {Director} — {Overview}"
    """
    df = df.copy()
    df["text_for_embedding"] = df.apply(
        lambda row: (
            f"{row['Series_Title']} ({row['Released_Year']}) — "
            f"{row['Genre']} — "
            f"Directed by {row['Director']} — "
            f"{row['Overview']}"
        ),
        axis=1,
    )
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline."""
    df = clean_released_year(df)
    df = clean_runtime(df)
    df = clean_gross(df)
    df = clean_numeric_columns(df)
    df = clean_genre(df)
    df = clean_cast(df)
    df = clean_director(df)
    df = build_text_representation(df)

    # Compute null rates for reporting
    total = len(df)
    gross_null_rate = df["Gross"].isna().sum() / total
    meta_null_rate = df["Meta_score"].isna().sum() / total

    print(f"[Preprocess] Loaded {total} movies")
    print(f"[Preprocess] Gross null rate: {gross_null_rate:.1%}")
    print(f"[Preprocess] Meta_score null rate: {meta_null_rate:.1%}")

    return df


def get_clean_dataframe(csv_path: str = None) -> pd.DataFrame:
    """Load and preprocess the dataset end-to-end."""
    from src.data_loader import load_raw_dataframe
    raw = load_raw_dataframe(csv_path)
    return preprocess(raw)

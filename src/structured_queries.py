"""
Structured Query Engine — deterministic pandas-based query functions.

Handles: title lookups, rankings, year/genre/threshold filters,
director aggregations, actor filtering, vote/gross queries.
"""
import pandas as pd
import numpy as np
from thefuzz import fuzz, process as fuzz_process


def _fuzzy_title_match(title: str, df: pd.DataFrame, threshold: int = 75) -> pd.DataFrame:
    """Find movies matching a title using fuzzy matching."""
    titles = df["Series_Title"].tolist()
    matches = fuzz_process.extract(title, titles, scorer=fuzz.token_sort_ratio, limit=5)
    matched_titles = [m[0] for m in matches if m[1] >= threshold]
    if not matched_titles:
        # try a lower threshold
        matches = fuzz_process.extract(title, titles, scorer=fuzz.partial_ratio, limit=5)
        matched_titles = [m[0] for m in matches if m[1] >= threshold]
    return df[df["Series_Title"].isin(matched_titles)]


def get_release_year(df: pd.DataFrame, title: str) -> dict:
    """Q1: Look up the release year for a specific movie title."""
    # Try exact match first
    exact = df[df["Series_Title"].str.lower() == title.lower()]
    if len(exact) > 0:
        row = exact.iloc[0]
        return {
            "title": row["Series_Title"],
            "released_year": int(row["Released_Year"]) if pd.notna(row["Released_Year"]) else None,
            "match_type": "exact",
        }

    # Try fuzzy match
    fuzzy = _fuzzy_title_match(title, df)
    if len(fuzzy) > 0:
        row = fuzzy.iloc[0]
        return {
            "title": row["Series_Title"],
            "released_year": int(row["Released_Year"]) if pd.notna(row["Released_Year"]) else None,
            "match_type": "fuzzy",
            "query_title": title,
        }

    return {"title": title, "released_year": None, "match_type": "not_found"}


def lookup_movie_info(df: pd.DataFrame, title: str, field: str = "release_year") -> dict:
    """General movie info lookup by title."""
    exact = df[df["Series_Title"].str.lower() == title.lower()]
    if len(exact) == 0:
        fuzzy = _fuzzy_title_match(title, df)
        if len(fuzzy) == 0:
            return {"title": title, "error": "Movie not found", "suggestion": _get_suggestions(title, df)}
        exact = fuzzy

    row = exact.iloc[0]
    field_map = {
        "release_year": ("Released_Year", lambda v: int(v) if pd.notna(v) else None),
        "rating": ("IMDB_Rating", lambda v: float(v) if pd.notna(v) else None),
        "overview": ("Overview", lambda v: str(v) if pd.notna(v) else None),
        "director": ("Director", lambda v: str(v) if pd.notna(v) else None),
        "genre": ("Genre", lambda v: str(v) if pd.notna(v) else None),
        "runtime": ("Runtime", lambda v: int(v) if pd.notna(v) else None),
        "meta_score": ("Meta_score", lambda v: float(v) if pd.notna(v) else None),
        "gross": ("Gross", lambda v: float(v) if pd.notna(v) else None),
    }

    if field in field_map:
        col, transform = field_map[field]
        return {
            "title": row["Series_Title"],
            "field": field,
            "value": transform(row[col]),
        }

    # Return all basic info
    return {
        "title": row["Series_Title"],
        "released_year": int(row["Released_Year"]) if pd.notna(row["Released_Year"]) else None,
        "rating": float(row["IMDB_Rating"]) if pd.notna(row["IMDB_Rating"]) else None,
        "director": row["Director"],
        "genre": row["Genre"],
        "runtime": int(row["Runtime"]) if pd.notna(row["Runtime"]) else None,
    }


def _get_suggestions(title: str, df: pd.DataFrame) -> list:
    """Get 'did you mean...' suggestions for misspelled titles."""
    titles = df["Series_Title"].tolist()
    matches = fuzz_process.extract(title, titles, scorer=fuzz.token_sort_ratio, limit=3)
    return [m[0] for m in matches if m[1] >= 50]


def top_movies_by_score(
    df: pd.DataFrame,
    year: int = None,
    metric: str = "Meta_score",
    n: int = 5,
    year_start: int = None,
    year_end: int = None,
    genre: str = None,
    meta_min: float = None,
    imdb_min: float = None,
    vote_min: int = None,
    gross_min: float = None,
    gross_max: float = None,
    sort_gross_asc: bool = False,
) -> dict:
    """
    Flexible top-N ranking with optional filters.
    Covers Q2, Q3, Q4, Q6 and variations.
    """
    filtered = df.copy()

    # Year filter
    if year is not None:
        filtered = filtered[filtered["Released_Year"] == year]
    if year_start is not None:
        filtered = filtered[filtered["Released_Year"] >= year_start]
    if year_end is not None:
        filtered = filtered[filtered["Released_Year"] <= year_end]

    # Genre filter
    if genre is not None:
        genre_lower = genre.lower().strip()
        filtered = filtered[
            filtered["genre_list"].apply(lambda gl: any(g.lower() == genre_lower for g in gl))
        ]

    # Threshold filters
    if meta_min is not None:
        filtered = filtered[filtered["Meta_score"] >= meta_min]
    if imdb_min is not None:
        filtered = filtered[filtered["IMDB_Rating"] >= imdb_min]
    if vote_min is not None:
        filtered = filtered[filtered["No_of_Votes"] >= vote_min]
    if gross_min is not None:
        filtered = filtered[filtered["has_gross"]]
        filtered = filtered[filtered["Gross"] >= gross_min]
    if gross_max is not None:
        filtered = filtered[filtered["has_gross"]]
        filtered = filtered[filtered["Gross"] <= gross_max]

    if len(filtered) == 0:
        return {"movies": [], "count": 0, "filters_applied": _describe_filters(locals())}

    # Sorting
    metric_col = "Meta_score" if metric.lower() in ("meta_score", "meta score", "metascore", "meta") else "IMDB_Rating"
    if sort_gross_asc:
        # Q6: sort by votes desc, then gross asc
        filtered = filtered.sort_values(
            by=["No_of_Votes", "Gross"], ascending=[False, True]
        )
    else:
        filtered = filtered.sort_values(by=metric_col, ascending=False)

    top = filtered.head(n)
    movies = []
    for _, row in top.iterrows():
        movies.append({
            "title": row["Series_Title"],
            "released_year": int(row["Released_Year"]) if pd.notna(row["Released_Year"]) else None,
            "genre": row["Genre"],
            "director": row["Director"],
            "imdb_rating": float(row["IMDB_Rating"]) if pd.notna(row["IMDB_Rating"]) else None,
            "meta_score": float(row["Meta_score"]) if pd.notna(row["Meta_score"]) else None,
            "no_of_votes": int(row["No_of_Votes"]) if pd.notna(row["No_of_Votes"]) else None,
            "gross": float(row["Gross"]) if pd.notna(row["Gross"]) else None,
            "overview": row["Overview"],
        })

    return {
        "movies": movies,
        "count": len(movies),
        "total_matching": len(filtered),
        "filters_applied": _describe_filters(locals()),
    }


def _describe_filters(params: dict) -> str:
    """Build a human-readable description of the filters applied."""
    parts = []
    if params.get("year"):
        parts.append(f"year={params['year']}")
    if params.get("year_start") or params.get("year_end"):
        ys = params.get("year_start", "...")
        ye = params.get("year_end", "...")
        parts.append(f"year range {ys}–{ye}")
    if params.get("genre"):
        parts.append(f"genre={params['genre']}")
    if params.get("meta_min"):
        parts.append(f"meta_score≥{params['meta_min']}")
    if params.get("imdb_min"):
        parts.append(f"imdb_rating≥{params['imdb_min']}")
    if params.get("vote_min"):
        parts.append(f"votes≥{params['vote_min']:,}")
    if params.get("gross_min"):
        parts.append(f"gross≥${params['gross_min']:,.0f}")
    return ", ".join(parts) if parts else "none"


def directors_by_repeated_gross_threshold(
    df: pd.DataFrame,
    gross_min: float = 500_000_000,
    min_count: int = 2,
) -> dict:
    """
    Q5: Find directors whose movies have grossed above a threshold at least N times.
    Returns each qualifying director with their highest-grossing movie.
    """
    # Filter movies with gross above threshold
    filtered = df[df["has_gross"] & (df["Gross"] >= gross_min)].copy()

    if len(filtered) == 0:
        # Try with a lower threshold — the dataset might use different scale
        return {
            "directors": [],
            "count": 0,
            "note": f"No movies found with Gross > ${gross_min:,.0f}. "
                    f"Max gross in dataset: ${df['Gross'].max():,.0f}",
        }

    # Group by director, filter groups with count >= min_count
    director_counts = filtered.groupby("Director").size()
    qualifying_directors = director_counts[director_counts >= min_count].index.tolist()

    results = []
    for director in qualifying_directors:
        director_movies = filtered[filtered["Director"] == director].sort_values(
            "Gross", ascending=False
        )
        top_movie = director_movies.iloc[0]
        results.append({
            "director": director,
            "qualifying_movie_count": int(director_counts[director]),
            "highest_grossing_movie": top_movie["Series_Title"],
            "highest_gross": float(top_movie["Gross"]),
            "all_qualifying_movies": [
                {
                    "title": row["Series_Title"],
                    "gross": float(row["Gross"]),
                    "year": int(row["Released_Year"]) if pd.notna(row["Released_Year"]) else None,
                }
                for _, row in director_movies.iterrows()
            ],
        })

    results.sort(key=lambda x: x["highest_gross"], reverse=True)

    return {
        "directors": results,
        "count": len(results),
        "threshold": gross_min,
        "min_count": min_count,
    }


def high_votes_low_gross(
    df: pd.DataFrame,
    vote_min: int = 1_000_000,
    n: int = 10,
) -> dict:
    """
    Q6: Top N movies with over vote_min votes but lower gross earnings.
    Sorted by votes desc, then gross asc (surfaces high-quality sleeper hits).
    """
    filtered = df[
        (df["No_of_Votes"] >= vote_min) & df["has_gross"]
    ].copy()

    filtered = filtered.sort_values(by="Gross", ascending=True)
    top = filtered.head(n)

    movies = []
    for _, row in top.iterrows():
        movies.append({
            "title": row["Series_Title"],
            "released_year": int(row["Released_Year"]) if pd.notna(row["Released_Year"]) else None,
            "genre": row["Genre"],
            "director": row["Director"],
            "imdb_rating": float(row["IMDB_Rating"]) if pd.notna(row["IMDB_Rating"]) else None,
            "no_of_votes": int(row["No_of_Votes"]) if pd.notna(row["No_of_Votes"]) else None,
            "gross": float(row["Gross"]) if pd.notna(row["Gross"]) else None,
        })

    return {
        "movies": movies,
        "count": len(movies),
        "vote_threshold": vote_min,
        "note": "Sorted by lowest gross earnings first (sleeper hits with high engagement).",
    }


def actor_movies_filtered(
    df: pd.DataFrame,
    actor: str,
    role_scope: str = "any",  # "lead" or "any"
    gross_min: float = None,
    imdb_min: float = None,
) -> dict:
    """
    Nice-to-have Q1: Filter movies by actor with optional gross and rating thresholds.
    Supports lead-only (Star1) or any role (Star1-4).
    """
    actor_lower = actor.lower().strip()

    if role_scope == "lead":
        filtered = df[df["lead_actor"].str.lower().str.strip() == actor_lower]
    else:
        filtered = df[
            df["cast_list"].apply(
                lambda cl: any(name.lower().strip() == actor_lower for name in cl)
            )
        ]

    if gross_min is not None:
        filtered = filtered[filtered["has_gross"] & (filtered["Gross"] >= gross_min)]
    if imdb_min is not None:
        filtered = filtered[filtered["IMDB_Rating"] >= imdb_min]

    filtered = filtered.sort_values("IMDB_Rating", ascending=False)

    movies = []
    for _, row in filtered.iterrows():
        movies.append({
            "title": row["Series_Title"],
            "released_year": int(row["Released_Year"]) if pd.notna(row["Released_Year"]) else None,
            "genre": row["Genre"],
            "director": row["Director"],
            "imdb_rating": float(row["IMDB_Rating"]) if pd.notna(row["IMDB_Rating"]) else None,
            "gross": float(row["Gross"]) if pd.notna(row["Gross"]) else None,
            "role": "Lead" if row["lead_actor"].lower().strip() == actor_lower else "Supporting",
        })

    return {
        "actor": actor,
        "role_scope": role_scope,
        "movies": movies,
        "count": len(movies),
    }


def movies_by_genre_and_thresholds(
    df: pd.DataFrame,
    genre: str,
    meta_min: float = None,
    imdb_min: float = None,
) -> dict:
    """Q4: Find movies by genre with meta score and/or IMDB rating thresholds."""
    return top_movies_by_score(
        df, genre=genre, meta_min=meta_min, imdb_min=imdb_min,
        n=100, metric="IMDB_Rating"
    )

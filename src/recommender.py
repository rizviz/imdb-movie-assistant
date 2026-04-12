"""
Recommendation Engine — suggests similar movies based on rating proximity.

Uses Euclidean distance on (IMDB_Rating, Meta_score) to find nearest neighbors.
"""
import pandas as pd
import numpy as np


def recommend_similar(
    df: pd.DataFrame,
    reference_movies: list[dict],
    n: int = 3,
    exclude_titles: list[str] = None,
) -> list[dict]:
    """
    Recommend movies similar to the given reference movies.
    
    Similarity is based on Euclidean distance of (IMDB_Rating, Meta_score).
    Excludes movies already in the result set.
    """
    if not reference_movies:
        return []

    exclude_titles = exclude_titles or []
    exclude_set = set(t.lower() for t in exclude_titles)

    # Get the reference point: average of reference movies' ratings
    ref_ratings = []
    ref_meta = []
    for movie in reference_movies:
        if movie.get("imdb_rating") is not None:
            ref_ratings.append(movie["imdb_rating"])
        if movie.get("meta_score") is not None:
            ref_meta.append(movie["meta_score"])

    if not ref_ratings:
        return []

    avg_rating = np.mean(ref_ratings)
    avg_meta = np.mean(ref_meta) if ref_meta else 70.0  # default if no meta scores

    # Filter to movies with both scores available
    candidates = df[
        df["IMDB_Rating"].notna() &
        df["Meta_score"].notna() &
        (~df["Series_Title"].str.lower().isin(exclude_set))
    ].copy()

    if len(candidates) == 0:
        return []

    # Compute Euclidean distance
    candidates = candidates.copy()
    candidates["_distance"] = np.sqrt(
        (candidates["IMDB_Rating"] - avg_rating) ** 2 +
        ((candidates["Meta_score"] - avg_meta) / 10) ** 2  # Scale meta_score down
    )

    closest = candidates.nsmallest(n, "_distance")

    recommendations = []
    for _, row in closest.iterrows():
        recommendations.append({
            "title": row["Series_Title"],
            "released_year": int(row["Released_Year"]) if pd.notna(row["Released_Year"]) else None,
            "imdb_rating": float(row["IMDB_Rating"]),
            "meta_score": float(row["Meta_score"]),
            "genre": row["Genre"],
        })

    return recommendations

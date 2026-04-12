"""
Semantic Query Engine — FAISS-based meaning-based retrieval over movie overviews.

Uses text-embedding-3-small embeddings and FAISS for local vector search.
Supports Q7 (comedies with death), Q8 (Spielberg sci-fi), Q9 (pre-1990 police).
"""
import os
import json
import numpy as np
import faiss
import pandas as pd
from src.llm_client import generate_embeddings_batch, generate_embedding
from src.utils import get_faiss_index_path


# Module-level cache
_index = None
_metadata = None


def build_index(df: pd.DataFrame, force_rebuild: bool = False) -> tuple:
    """
    Build (or load from disk) the FAISS index for semantic search.
    
    Each document combines title, year, genre, director, and overview
    for richer semantic matching.
    """
    index_dir = get_faiss_index_path()
    index_path = os.path.join(index_dir, "index.faiss")
    meta_path = os.path.join(index_dir, "metadata.json")

    if not force_rebuild and os.path.exists(index_path) and os.path.exists(meta_path):
        print("[Semantic] Loading existing FAISS index from disk...")
        index = faiss.read_index(index_path)
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        return index, metadata

    print("[Semantic] Building new FAISS index...")
    texts = df["text_for_embedding"].tolist()

    # Build metadata for post-retrieval filtering
    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            "title": row["Series_Title"],
            "released_year": int(row["Released_Year"]) if pd.notna(row["Released_Year"]) else None,
            "genre": row["Genre"],
            "genre_list": row["genre_list"],
            "director": row["Director"],
            "overview": row["Overview"] if pd.notna(row["Overview"]) else "",
            "imdb_rating": float(row["IMDB_Rating"]) if pd.notna(row["IMDB_Rating"]) else None,
            "meta_score": float(row["Meta_score"]) if pd.notna(row["Meta_score"]) else None,
        })

    # Generate embeddings
    print(f"[Semantic] Generating embeddings for {len(texts)} documents...")
    embeddings = generate_embeddings_batch(texts)
    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Build FAISS index (L2 distance, then we'll use inner product after normalization)
    dimension = embeddings_array.shape[1]
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_array)
    
    index = faiss.IndexFlatIP(dimension)  # Inner product after normalization = cosine similarity
    index.add(embeddings_array)

    # Save to disk
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    print(f"[Semantic] Index built with {index.ntotal} vectors of dimension {dimension}")
    return index, metadata


def get_index_and_metadata(df: pd.DataFrame = None) -> tuple:
    """Get the cached index and metadata, building if necessary."""
    global _index, _metadata
    if _index is None:
        if df is None:
            raise RuntimeError("Must provide DataFrame on first call to build the index.")
        _index, _metadata = build_index(df)
    return _index, _metadata


def semantic_search(
    query: str,
    df: pd.DataFrame = None,
    top_k: int = 10,
    genre_filter: str = None,
    year_before: int = None,
    year_after: int = None,
    director_filter: str = None,
) -> list[dict]:
    """
    Search movie plots by meaning with optional post-retrieval filtering.
    
    Returns list of dicts with title, overview, score, and metadata.
    """
    index, metadata = get_index_and_metadata(df)

    # Generate query embedding
    query_embedding = np.array([generate_embedding(query)], dtype=np.float32)
    faiss.normalize_L2(query_embedding)

    # Search with a larger k to account for post-filtering
    search_k = min(top_k * 5, len(metadata))
    scores, indices = index.search(query_embedding, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = metadata[idx]

        # Apply post-retrieval filters
        if genre_filter:
            genre_lower = genre_filter.lower().strip()
            if not any(g.lower() == genre_lower for g in item.get("genre_list", [])):
                continue

        if year_before is not None and item.get("released_year"):
            if item["released_year"] >= year_before:
                continue

        if year_after is not None and item.get("released_year"):
            if item["released_year"] <= year_after:
                continue

        if director_filter:
            if item.get("director", "").lower().strip() != director_filter.lower().strip():
                continue

        results.append({
            "title": item["title"],
            "released_year": item.get("released_year"),
            "genre": item.get("genre"),
            "director": item.get("director"),
            "overview": item.get("overview"),
            "imdb_rating": item.get("imdb_rating"),
            "meta_score": item.get("meta_score"),
            "similarity_score": float(score),
        })

        if len(results) >= top_k:
            break

    return results


def search_comedies_with_death(df: pd.DataFrame = None, top_k: int = 10) -> list[dict]:
    """Q7: Find comedy movies where death or dead people are involved in the plot."""
    results = semantic_search(
        query="comedy movie where someone dies, death, dead people, murder, killing, funeral",
        df=df,
        top_k=top_k,
        genre_filter="Comedy",
    )
    return results


def search_spielberg_scifi(df: pd.DataFrame = None) -> list[dict]:
    """
    Q8: Spielberg's sci-fi movies. 
    This is actually a structured filter + LLM summarization (hybrid).
    """
    results = semantic_search(
        query="Steven Spielberg science fiction movie futuristic aliens space technology",
        df=df,
        top_k=20,
        director_filter="Steven Spielberg",
        genre_filter="Sci-Fi",
    )
    return results


def search_police_pre1990(df: pd.DataFrame = None, top_k: int = 15) -> list[dict]:
    """
    Q9: Movies before 1990 with police involvement (similarity-based, not keyword).
    Search for related concepts: law enforcement, detective, sheriff, cops, investigation.
    """
    results = semantic_search(
        query="police law enforcement detective investigation cops sheriff officer crime solving pursuit arrest",
        df=df,
        top_k=top_k,
        year_before=1990,
    )
    return results

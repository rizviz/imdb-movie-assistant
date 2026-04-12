"""
Response Builder — formats query results into readable responses.

Adds reasoning traces, query-path labels, and optional recommendations.
"""
import pandas as pd
from src.utils import format_currency, format_number
from src.recommender import recommend_similar


def build_response(
    result: dict,
    query_type: str,
    df: pd.DataFrame = None,
    original_query: str = "",
) -> str:
    """
    Build a formatted response from raw query results.
    
    query_type: "structured", "semantic", "hybrid", "clarification", "lookup"
    """
    parts = []

    if query_type == "lookup":
        parts.append(_format_lookup(result))
    elif query_type == "structured":
        parts.append(_format_structured(result))
    elif query_type == "semantic":
        parts.append(_format_semantic(result))
    elif query_type == "hybrid":
        parts.append(_format_hybrid(result))
    elif query_type == "clarification":
        return _format_clarification(result)
    elif query_type == "director_aggregation":
        parts.append(_format_director_aggregation(result))
    elif query_type == "actor_filter":
        parts.append(_format_actor_filter(result))
    else:
        parts.append(str(result))

    # Add reasoning trace
    reasoning = _build_reasoning(result, query_type, original_query)
    if reasoning:
        parts.append(f"\n---\n**🔍 Reasoning:** {reasoning}")

    # Add query path label
    path_label = _get_path_label(query_type)
    parts.append(f"\n*{path_label}*")

    # Add recommendations if we have movies and a dataframe
    if df is not None and "movies" in result and result.get("movies"):
        exclude = [m["title"] for m in result["movies"]]
        recs = recommend_similar(df, result["movies"], n=3, exclude_titles=exclude)
        if recs:
            rec_text = ", ".join(
                f"**{r['title']}** ({r.get('released_year', 'N/A')}) — "
                f"IMDB {r['imdb_rating']}, Meta {r['meta_score']}"
                for r in recs
            )
            parts.append(f"\n💡 **You might also like:** {rec_text}")

    return "\n".join(parts)


def _format_lookup(result: dict) -> str:
    """Format a single movie lookup result."""
    if result.get("error"):
        msg = f"❌ {result['error']}"
        if result.get("suggestion"):
            msg += f"\n\nDid you mean: {', '.join(result['suggestion'])}?"
        return msg

    title = result.get("title", "Unknown")
    field = result.get("field", "")
    value = result.get("value")

    if "released_year" in result:
        return f"🎬 **{title}** was released in **{result['released_year']}**."

    if field == "release_year":
        return f"🎬 **{title}** was released in **{value}**."
    elif field == "rating":
        return f"🎬 **{title}** has an IMDB rating of **{value}**."
    elif field == "overview":
        return f"🎬 **{title}**\n\n{value}"
    elif field == "director":
        return f"🎬 **{title}** was directed by **{value}**."
    else:
        parts = [f"🎬 **{title}**"]
        if result.get("released_year"):
            parts.append(f"- Released: {result['released_year']}")
        if result.get("rating"):
            parts.append(f"- IMDB Rating: {result['rating']}")
        if result.get("director"):
            parts.append(f"- Director: {result['director']}")
        if result.get("genre"):
            parts.append(f"- Genre: {result['genre']}")
        return "\n".join(parts)


def _format_structured(result: dict) -> str:
    """Format structured query results as a markdown table."""
    movies = result.get("movies", [])
    if not movies:
        return "No movies matched your criteria."

    count = result.get("count", len(movies))
    total = result.get("total_matching", count)

    header = f"Found **{total}** matching movies"
    if total > count:
        header += f" (showing top {count})"
    header += ":\n"

    # Build markdown table
    table = "| # | Title | Year | IMDB | Meta | Genre |\n"
    table += "|---|-------|------|------|------|-------|\n"
    for i, m in enumerate(movies, 1):
        title = m.get("title", "Unknown")
        year = m.get("released_year", "N/A")
        imdb = m.get("imdb_rating", "N/A")
        meta = m.get("meta_score", "N/A")
        genre = m.get("genre", "N/A")
        gross = m.get("gross")
        gross_str = f" | {format_currency(gross)}" if gross else ""
        table += f"| {i} | {title} | {year} | {imdb} | {meta} | {genre} |\n"

    # If any movies have gross info, show it
    has_gross = any(m.get("gross") is not None for m in movies)
    if has_gross:
        table += "\n**Gross Earnings:**\n"
        for i, m in enumerate(movies, 1):
            if m.get("gross") is not None:
                table += f"- {m['title']}: {format_currency(m['gross'])}\n"

    # If any movies have vote info, show it
    has_votes = any(m.get("no_of_votes") is not None for m in movies)
    if has_votes and not has_gross:
        table += "\n**Vote Counts:**\n"
        for i, m in enumerate(movies, 1):
            if m.get("no_of_votes") is not None:
                table += f"- {m['title']}: {format_number(m['no_of_votes'])} votes\n"

    note = result.get("note", "")
    if note:
        table += f"\n> ℹ️ {note}"

    return header + table


def _format_semantic(result: dict) -> str:
    """Format semantic search results."""
    if isinstance(result, list):
        movies = result
    else:
        movies = result.get("movies", result if isinstance(result, list) else [])

    if not movies:
        return ("No movies matched your search based on plot similarity. "
                "Try rephrasing your query.")

    header = f"Found **{len(movies)}** movies based on plot similarity:\n\n"
    lines = []
    for i, m in enumerate(movies, 1):
        title = m.get("title", "Unknown")
        year = m.get("released_year", "N/A")
        genre = m.get("genre", "N/A")
        score = m.get("similarity_score", 0)
        overview = m.get("overview", "")
        if len(overview) > 200:
            overview = overview[:200] + "..."
        lines.append(
            f"**{i}. {title}** ({year}) — {genre}\n"
            f"   *Similarity: {score:.2f}*\n"
            f"   {overview}\n"
        )

    footer = ("\n> ⚠️ These results are based on plot similarity and may not be exact matches.")
    return header + "\n".join(lines) + footer


def _format_hybrid(result: dict) -> str:
    """Format hybrid (structured filter + semantic/LLM) results."""
    if isinstance(result, str):
        return result  # Already formatted (e.g., LLM summary)

    # If it's a list of movies with a summary
    if isinstance(result, dict) and "summary" in result:
        movies = result.get("movies", [])
        summary = result.get("summary", "")

        header = f"Found **{len(movies)}** matching movies:\n\n"
        movie_list = ""
        for i, m in enumerate(movies, 1):
            title = m.get("title", "Unknown")
            year = m.get("released_year", "N/A")
            movie_list += f"**{i}. {title}** ({year})\n"

        return header + movie_list + f"\n**Plot Summaries:**\n\n{summary}"

    return _format_semantic(result)


def _format_director_aggregation(result: dict) -> str:
    """Format director aggregation results (Q5)."""
    directors = result.get("directors", [])
    if not directors:
        note = result.get("note", "No directors matched the criteria.")
        return f"❌ {note}"

    header = (
        f"Found **{len(directors)}** directors with movies grossing over "
        f"{format_currency(result.get('threshold', 0))} at least "
        f"{result.get('min_count', 2)} times:\n\n"
    )

    lines = []
    for d in directors:
        director = d["director"]
        count = d["qualifying_movie_count"]
        top_movie = d["highest_grossing_movie"]
        top_gross = format_currency(d["highest_gross"])

        lines.append(f"### 🎬 {director} ({count} qualifying films)")
        lines.append(f"**Highest grossing:** {top_movie} ({top_gross})\n")

        if d.get("all_qualifying_movies"):
            for movie in d["all_qualifying_movies"]:
                lines.append(
                    f"- {movie['title']} ({movie.get('year', 'N/A')}) — "
                    f"{format_currency(movie['gross'])}"
                )
        lines.append("")

    return header + "\n".join(lines)


def _format_actor_filter(result: dict) -> str:
    """Format actor-filtered movie results."""
    actor = result.get("actor", "Unknown")
    scope = result.get("role_scope", "any")
    movies = result.get("movies", [])

    scope_desc = "as lead actor" if scope == "lead" else "in any role"

    if not movies:
        return f"No movies found for **{actor}** ({scope_desc}) matching your criteria."

    header = f"**{actor}** movies ({scope_desc}) — {len(movies)} results:\n\n"

    table = "| # | Title | Year | IMDB | Gross | Role |\n"
    table += "|---|-------|------|------|-------|------|\n"
    for i, m in enumerate(movies, 1):
        table += (
            f"| {i} | {m['title']} | {m.get('released_year', 'N/A')} | "
            f"{m.get('imdb_rating', 'N/A')} | {format_currency(m.get('gross'))} | "
            f"{m.get('role', 'N/A')} |\n"
        )

    return header + table


def _format_clarification(result: dict) -> str:
    """Format a clarification request."""
    question = result.get("question", "Could you clarify your question?")
    options = result.get("options", [])

    msg = f"🤔 **Clarification needed:**\n\n{question}"
    if options:
        msg += "\n\nOptions:\n"
        for i, opt in enumerate(options, 1):
            msg += f"{i}. {opt}\n"

    return msg


def _build_reasoning(result: dict, query_type: str, original_query: str) -> str:
    """Build a reasoning trace for the response."""
    if query_type == "lookup":
        match_type = result.get("match_type", "exact")
        if match_type == "fuzzy":
            return f'Matched "{result.get("query_title")}" to "{result.get("title")}" using fuzzy title matching.'
        return f'Found exact match for "{result.get("title")}" in the dataset.'

    if query_type == "structured":
        filters = result.get("filters_applied", "none")
        total = result.get("total_matching", 0)
        return f"Filtered 1,000 movies with criteria ({filters}). Found {total} matching results, ranked by score."

    if query_type == "semantic":
        count = len(result) if isinstance(result, list) else result.get("count", 0)
        return (
            f"Performed semantic similarity search on movie plot overviews. "
            f"Found {count} relevant matches using embedding-based retrieval."
        )

    if query_type == "hybrid":
        return (
            "Combined structured filtering (director/genre) with semantic retrieval "
            "and LLM summarization to provide plot summaries."
        )

    if query_type == "director_aggregation":
        threshold = result.get("threshold", 0)
        min_count = result.get("min_count", 2)
        return (
            f"Filtered movies with gross > {format_currency(threshold)}, "
            f"grouped by director, and selected directors with ≥{min_count} qualifying films."
        )

    if query_type == "actor_filter":
        actor = result.get("actor", "")
        scope = result.get("role_scope", "any")
        return (
            f'Filtered movies featuring {actor} ({scope} role scope) '
            f'with the specified rating and earnings thresholds.'
        )

    return ""


def _get_path_label(query_type: str) -> str:
    """Return the query path label for display."""
    labels = {
        "lookup": "📋 Answered via structured query",
        "structured": "📋 Answered via structured query",
        "semantic": "🔍 Answered via semantic retrieval",
        "hybrid": "🔄 Answered via hybrid flow (structured filter + semantic/LLM)",
        "clarification": "❓ Awaiting clarification",
        "director_aggregation": "📋 Answered via structured query (director aggregation)",
        "actor_filter": "📋 Answered via structured query (actor filter)",
    }
    return labels.get(query_type, "📋 Answered via structured query")

"""
Query Orchestration Router — uses OpenAI function-calling to route queries.

Routing paths:
- Structured: exact lookup, ranking, filtering, aggregation
- Semantic: meaning-based plot retrieval
- Hybrid: semantic retrieval + deterministic filters
- Clarification: ambiguous intent needing follow-up
"""
import json
import pandas as pd
from src.llm_client import chat_completion, summarize_plots
from src.structured_queries import (
    get_release_year,
    lookup_movie_info,
    top_movies_by_score,
    directors_by_repeated_gross_threshold,
    high_votes_low_gross,
    actor_movies_filtered,
    movies_by_genre_and_thresholds,
)
from src.semantic_queries import (
    semantic_search,
    search_comedies_with_death,
    search_spielberg_scifi,
    search_police_pre1990,
)
from src.response_builder import build_response


# ─────────────────────────────────────────────
# Tool definitions for OpenAI function-calling
# ─────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_movie_info",
            "description": "Look up release year, rating, overview, director, or other basic info for a specific movie title. Use this for questions about a single movie.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The movie title to look up",
                    },
                    "field": {
                        "type": "string",
                        "enum": ["release_year", "rating", "overview", "director", "genre", "runtime", "meta_score", "gross", "all"],
                        "description": "Which field to retrieve. Use 'all' for general info.",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "top_movies_ranked",
            "description": "Get top-N movies ranked by IMDB rating or meta score, with optional filters for genre, year range, rating thresholds, vote count, and gross earnings. Use for any ranking or 'top N' question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of top movies to return",
                        "default": 10,
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["IMDB_Rating", "Meta_score"],
                        "description": "Which metric to rank by",
                        "default": "IMDB_Rating",
                    },
                    "year": {
                        "type": "integer",
                        "description": "Exact year to filter by (use year_start/year_end for ranges)",
                    },
                    "year_start": {
                        "type": "integer",
                        "description": "Start of year range (inclusive)",
                    },
                    "year_end": {
                        "type": "integer",
                        "description": "End of year range (inclusive)",
                    },
                    "genre": {
                        "type": "string",
                        "description": "Genre to filter by (e.g., 'Comedy', 'Horror', 'Sci-Fi', 'Action')",
                    },
                    "meta_min": {
                        "type": "number",
                        "description": "Minimum meta score threshold",
                    },
                    "imdb_min": {
                        "type": "number",
                        "description": "Minimum IMDB rating threshold",
                    },
                    "vote_min": {
                        "type": "integer",
                        "description": "Minimum number of votes",
                    },
                    "gross_min": {
                        "type": "number",
                        "description": "Minimum gross earnings in USD",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_plot_search",
            "description": "Search movie plots by meaning/theme for questions about movie content, themes, or plot elements. Use when the question is about what happens in movies (e.g., 'movies with death', 'movies about police', 'movies involving time travel'). Do NOT use for simple ranking or filtering questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of the plot theme or content to search for",
                    },
                    "genre_filter": {
                        "type": "string",
                        "description": "Optional genre to filter results (e.g., 'Comedy', 'Horror')",
                    },
                    "year_before": {
                        "type": "integer",
                        "description": "Only return movies released before this year",
                    },
                    "year_after": {
                        "type": "integer",
                        "description": "Only return movies released after this year",
                    },
                    "director_filter": {
                        "type": "string",
                        "description": "Optional director name to filter results",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "director_aggregation",
            "description": "Find directors meeting specific criteria across their filmography, such as directors with multiple movies above a gross earnings threshold. Use for questions like 'directors with highest grossing movies' or 'directors who have grossed over X at least N times'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gross_min": {
                        "type": "number",
                        "description": "Minimum gross earnings threshold in USD. Note: values in the dataset are raw USD (e.g., 500000000 for $500M).",
                    },
                    "min_count": {
                        "type": "integer",
                        "description": "Minimum number of movies meeting the threshold",
                        "default": 2,
                    },
                },
                "required": ["gross_min"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "high_votes_low_gross",
            "description": "Find movies with high vote counts but relatively lower gross earnings (sleeper hits). Use for questions about popular movies that didn't earn as much commercially.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vote_min": {
                        "type": "integer",
                        "description": "Minimum number of votes",
                        "default": 1000000,
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "actor_movies_filtered",
            "description": "Find movies for a specific actor with optional gross earnings and IMDB rating filters. Can filter by lead role only or any role. Use when the question mentions a specific actor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actor": {
                        "type": "string",
                        "description": "Actor name to search for",
                    },
                    "role_scope": {
                        "type": "string",
                        "enum": ["lead", "any"],
                        "description": "Whether to search for lead roles only (Star1) or any role (Star1-4)",
                    },
                    "gross_min": {
                        "type": "number",
                        "description": "Minimum gross earnings in USD",
                    },
                    "imdb_min": {
                        "type": "number",
                        "description": "Minimum IMDB rating",
                    },
                },
                "required": ["actor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_movie_plots",
            "description": "Summarize movie plots for a director's movies in a specific genre. Use for questions about plot summaries, especially when combined with director and genre filters. This is a hybrid operation: structured filter + LLM summarization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "director": {
                        "type": "string",
                        "description": "Director name",
                    },
                    "genre": {
                        "type": "string",
                        "description": "Genre to filter by",
                    },
                    "query_context": {
                        "type": "string",
                        "description": "Additional context about what to summarize",
                    },
                },
                "required": ["director", "genre"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_clarification",
            "description": "Ask the user a clarifying question before executing a query. Use when the user's intent is ambiguous, such as when asking about an actor without specifying lead vs any role.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The clarifying question to ask",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of choices for the user",
                    },
                },
                "required": ["question"],
            },
        },
    },
]


def route_query(
    user_message: str,
    df: pd.DataFrame,
    conversation_history: list = None,
    pending_clarification: dict = None,
) -> tuple:
    """
    Route a user query to the appropriate handler using function-calling.
    
    Returns: (response_text, query_type, raw_result)
    """
    if conversation_history is None:
        conversation_history = []

    # If there's a pending clarification, handle the response
    if pending_clarification:
        return _handle_clarification_response(
            user_message, df, pending_clarification
        )

    # Build message list for the router
    system_msg = {
        "role": "system",
        "content": (
            "You are a movie database assistant. Route user questions to the appropriate "
            "function to answer them. The dataset contains the top 1000 IMDB movies.\n\n"
            "Routing guidelines:\n"
            "- For questions about a specific movie's year, rating, etc: use lookup_movie_info\n"
            "- For 'top N' rankings with filters: use top_movies_ranked\n"
            "- For plot/theme-based questions (death in comedies, police involvement): use semantic_plot_search\n"
            "- For director filmography aggregations (grossing threshold): use director_aggregation\n"
            "- For high-votes-low-earnings questions: use high_votes_low_gross\n"
            "- For actor-specific queries with filters: use actor_movies_filtered\n"
            "  - If the user asks about an actor without specifying lead vs any role, "
            "    AND also has gross or rating filters, use ask_clarification first\n"
            "- For plot summaries (e.g., 'summarize Spielberg sci-fi plots'): use summarize_movie_plots\n"
            "- For ambiguous questions: use ask_clarification\n\n"
            "Important notes:\n"
            "- Gross values in the dataset are raw USD (e.g., 500000000 for $500M, 50000000 for $50M)\n"
            "- 'Meta score' = Meta_score column\n"
            "- 'IMDB rating' = IMDB_Rating column\n"
            "- For comedy movies with death/dead people: use semantic_plot_search with genre_filter='Comedy'\n"
            "- For 'before 1990' police movies: use semantic_plot_search with year_before=1990\n"
        ),
    }

    messages = [system_msg] + conversation_history + [{"role": "user", "content": user_message}]

    # Call the LLM with function-calling
    response = chat_completion(messages, tools=TOOLS, tool_choice="auto")

    # Check if the model wants to call a function
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        print(f"[Router] Selected tool: {function_name}")
        print(f"[Router] Arguments: {json.dumps(arguments, indent=2)}")

        return _execute_tool(function_name, arguments, df, user_message)

    # If no function call, return the direct response
    return (response.content or "I'm not sure how to answer that. Could you rephrase?", "direct", {})


def _execute_tool(
    function_name: str,
    arguments: dict,
    df: pd.DataFrame,
    original_query: str,
) -> tuple:
    """Execute the selected tool and build a response."""

    if function_name == "lookup_movie_info":
        title = arguments.get("title", "")
        field = arguments.get("field", "release_year")
        result = lookup_movie_info(df, title, field)
        response = build_response(result, "lookup", df, original_query)
        return (response, "lookup", result)

    elif function_name == "top_movies_ranked":
        result = top_movies_by_score(
            df,
            year=arguments.get("year"),
            metric=arguments.get("metric", "IMDB_Rating"),
            n=arguments.get("n", 10),
            year_start=arguments.get("year_start"),
            year_end=arguments.get("year_end"),
            genre=arguments.get("genre"),
            meta_min=arguments.get("meta_min"),
            imdb_min=arguments.get("imdb_min"),
            vote_min=arguments.get("vote_min"),
            gross_min=arguments.get("gross_min"),
        )
        response = build_response(result, "structured", df, original_query)
        return (response, "structured", result)

    elif function_name == "semantic_plot_search":
        results = semantic_search(
            query=arguments.get("query", ""),
            df=df,
            top_k=arguments.get("top_k", 10),
            genre_filter=arguments.get("genre_filter"),
            year_before=arguments.get("year_before"),
            year_after=arguments.get("year_after"),
            director_filter=arguments.get("director_filter"),
        )
        result = {"movies": results, "count": len(results)}
        response = build_response(result, "semantic", df, original_query)
        return (response, "semantic", result)

    elif function_name == "director_aggregation":
        result = directors_by_repeated_gross_threshold(
            df,
            gross_min=arguments.get("gross_min", 500_000_000),
            min_count=arguments.get("min_count", 2),
        )
        response = build_response(result, "director_aggregation", df, original_query)
        return (response, "director_aggregation", result)

    elif function_name == "high_votes_low_gross":
        result = high_votes_low_gross(
            df,
            vote_min=arguments.get("vote_min", 1_000_000),
            n=arguments.get("n", 10),
        )
        response = build_response(result, "structured", df, original_query)
        return (response, "structured", result)

    elif function_name == "actor_movies_filtered":
        # Check if we need clarification for role scope
        actor = arguments.get("actor", "")
        role_scope = arguments.get("role_scope")
        gross_min = arguments.get("gross_min")
        imdb_min = arguments.get("imdb_min")

        if role_scope is None and (gross_min or imdb_min):
            # Trigger clarification
            result = {
                "question": (
                    f'Are you looking for movies where {actor} is the lead actor (Star1), '
                    f'or any movie where {actor} has appeared (lead or supporting)?'
                ),
                "options": [
                    f"Movies where {actor} is the lead actor",
                    f"Any movie featuring {actor} (lead or supporting)",
                ],
                "pending_function": "actor_movies_filtered",
                "pending_args": arguments,
            }
            response = build_response(result, "clarification", df=None)
            return (response, "clarification", result)

        result = actor_movies_filtered(
            df,
            actor=actor,
            role_scope=role_scope or "any",
            gross_min=gross_min,
            imdb_min=imdb_min,
        )
        response = build_response(result, "actor_filter", df, original_query)
        return (response, "actor_filter", result)

    elif function_name == "summarize_movie_plots":
        director = arguments.get("director", "")
        genre = arguments.get("genre", "")
        context = arguments.get("query_context", "")

        # Get movies using semantic search with director + genre filter
        results = semantic_search(
            query=f"{director} {genre} movies",
            df=df,
            top_k=20,
            director_filter=director,
            genre_filter=genre,
        )

        if not results:
            # Fallback to structured filter
            genre_lower = genre.lower().strip()
            filtered = df[
                (df["Director"].str.lower().str.strip() == director.lower().strip()) &
                (df["genre_list"].apply(lambda gl: any(g.lower() == genre_lower for g in gl)))
            ]
            results = [
                {
                    "title": row["Series_Title"],
                    "released_year": int(row["Released_Year"]) if pd.notna(row["Released_Year"]) else None,
                    "overview": row["Overview"],
                    "imdb_rating": float(row["IMDB_Rating"]) if pd.notna(row["IMDB_Rating"]) else None,
                }
                for _, row in filtered.iterrows()
            ]

        if not results:
            return (
                f"No {genre} movies by {director} found in the dataset.",
                "hybrid",
                {"movies": [], "count": 0},
            )

        # Sort by rating and summarize
        results.sort(key=lambda x: x.get("imdb_rating", 0) or 0, reverse=True)

        plots = [
            {"title": r["title"], "year": r.get("released_year"), "overview": r.get("overview", "")}
            for r in results
        ]

        summary = summarize_plots(
            plots,
            context=f"These are {director}'s top-rated {genre} movies. {context}",
        )

        result = {"movies": results, "summary": summary, "count": len(results)}
        response = build_response(result, "hybrid", df, original_query)
        return (response, "hybrid", result)

    elif function_name == "ask_clarification":
        result = {
            "question": arguments.get("question", "Could you clarify?"),
            "options": arguments.get("options", []),
        }
        response = build_response(result, "clarification")
        return (response, "clarification", result)

    else:
        return (f"Unknown function: {function_name}", "error", {})


def _handle_clarification_response(
    user_message: str,
    df: pd.DataFrame,
    pending: dict,
) -> tuple:
    """Handle user's response to a clarification question."""
    function_name = pending.get("pending_function", "")
    args = pending.get("pending_args", {})

    if function_name == "actor_movies_filtered":
        # Determine role scope from user's response
        msg_lower = user_message.lower()
        if any(word in msg_lower for word in ["lead", "star1", "main", "first", "1", "lead actor"]):
            args["role_scope"] = "lead"
        else:
            args["role_scope"] = "any"

        return _execute_tool(function_name, args, df, user_message)

    return ("I'm sorry, I couldn't process your clarification. Please try asking your question again.", "error", {})

Apr-12-2026, 10:36 AM CST

## MVP dev plan — v2

### MVP objective

Build a local Streamlit-based conversational movie assistant that uses the IMDB dataset to answer structured and semantic questions reliably.

The MVP ends when the app can run locally, answer the core assignment questions, and support a basic conversational flow with clarification where needed.

### Model choices

Pin these up front so cost, latency, and capability are predictable.

| Role | Model | Rationale |
|------|-------|-----------|
| Routing / orchestration | `gpt-4o-mini` | Fast, cheap, strong at function-calling |
| Summarization / response | `gpt-4o-mini` | Sufficient for narrative answers over retrieved data |
| Embeddings | `text-embedding-3-small` | 1536-dim, low cost, good semantic quality for plot retrieval |

If response quality on summarization tasks (e.g., Spielberg plot summaries) feels thin, upgrade that single call to `gpt-4o`. Keep everything else on `gpt-4o-mini` to stay fast and cheap.

### Core MVP scope

The MVP should include only these core capabilities:

1. dataset ingestion and preprocessing
2. structured query engine
3. semantic retrieval over plot summaries
4. query orchestration via function-calling
5. response construction with reasoning traces
6. Streamlit UI
7. basic quality checks
8. lightweight recommendation engine (nice-to-have, low effort)

Voice, Docker packaging, and extra polish should stay outside the MVP unless the core is already stable. Voice is addressed in the post-MVP section below.

## Workstream 1: repository and environment setup

Set up the project skeleton first so the rest of the work has a clean home.

Create:

* root project folder
* `src/`
* `data/`
* `vectorstore/`
* `tests/`
* `app.py`
* `requirements.txt`
* `.env.example`
* `README.md`

Set up:

* Python virtual environment
* OpenAI API key loading through `.env`
* dataset unzip/load flow
* basic Streamlit launch

Exit condition:
the app starts, the dataset can be loaded, and the repo layout is stable.

## Workstream 2: data pipeline

This is the foundation. Do this before touching the LLM.

### General tasks

* load CSV from the zip
* inspect column quality and nulls
* normalize `Released_Year` — strip non-numeric entries (e.g., "PG"), cast to int, handle nulls
* convert `IMDB_Rating`, `Meta_score`, `No_of_votes` into numeric form

### Gross column — requires special handling

The `Gross` field is the most problematic column in this dataset:

* values are quoted strings with embedded commas: `"28,341,469"`
* many rows have null/missing gross values
* no currency symbol, but values are in USD

Cleaning steps:

* strip commas and quotes, cast to float
* preserve NaN for missing values — do not fill with zero (that would corrupt rankings and threshold filters)
* add a `has_gross` boolean column for queries that need to filter on gross availability
* document the null rate so response builder can note when results may be incomplete

### Genre column — multi-value field

`Genre` contains comma-separated values within a single cell: `"Crime, Drama"`.

Approach:

* keep the original `Genre` column as-is for display
* add a `genre_list` column: split on `, ` and store as a Python list
* all genre filtering should use `genre_list.apply(lambda g: target in g)` or equivalent set-membership check
* do not explode into rows — that inflates the dataframe and complicates aggregations

### Cast normalization

* combine `Star1` through `Star4` into a `cast_list` column (list of strings)
* add a `lead_actor` alias pointing to `Star1` for the Al Pacino clarification flow
* strip whitespace from all names

### Runtime normalization

* strip ` min` suffix, cast to int

### Output

* one cleaned dataframe for structured querying
* one text-ready representation for semantic indexing (format: `"{title} ({year}) — {genre} — Directed by {director} — {overview}"`)

Exit condition:
a single clean dataframe is available and can support exact filters and rankings without ad hoc fixes later. Gross nulls, multi-genre values, and cast lookups all work correctly.

## Workstream 3: structured query engine

Build deterministic functions for exact questions.

This layer should handle:

* title lookups (exact and fuzzy)
* release year lookups
* top-N ranking by score (IMDB or Meta)
* year range filters
* genre filters (using `genre_list` membership)
* rating and meta-score thresholds
* vote-count and gross filters (with null-awareness)
* director aggregations (including "at least N times" constraints)
* actor-based filtering (lead vs. any role, using `lead_actor` and `cast_list`)

### Representative functions mapped to assignment questions

| Assignment Q | Function signature sketch |
|---|---|
| Q1: When did The Matrix release? | `get_release_year(title: str)` |
| Q2: Top 5 movies of 2019 by meta score | `top_movies_by_score(year, metric, n)` |
| Q3: Top 7 comedy movies 2010–2020 by IMDB | `top_movies_by_genre_year_range(genre, start, end, metric, n)` |
| Q4: Top horror, meta > 85 and IMDB > 8 | `movies_by_genre_and_thresholds(genre, meta_min, imdb_min)` |
| Q5: Directors with gross > 500M at least twice | `directors_by_repeated_gross_threshold(gross_min, min_count)` |
| Q6: Top 10 movies, > 1M votes, lower gross | `high_votes_low_gross(vote_min, n)` — sort by votes desc, then gross asc |
| Nice-to-have 1: Al Pacino > $50M and IMDB ≥ 8 | `actor_movies_filtered(actor, role_scope, gross_min, imdb_min)` |

This layer should return raw structured results (dataframes or dicts), not LLM prose.

### Q5 implementation note

Question 5 is the hardest structured query. It requires:

1. filter movies with `Gross > 500_000_000` (note: dataset gross values are raw USD, not millions)
2. group by `Director`
3. filter groups where count ≥ 2
4. for each qualifying director, return their highest-grossing movie

Verify the 500M threshold against actual data — the dataset may use raw values or abbreviated values. Check the Gross column max to calibrate.

### Q6 interpretation note

"Lower gross earnings" is ambiguous. Implement as: movies with > 1M votes, sorted by gross ascending (lowest gross first), take top 10. This surfaces high-quality sleeper hits.

Exit condition:
all exact and ranking-style assignment questions can be answered directly from pandas, including Q5's aggregation constraint.

## Workstream 4: semantic retrieval layer

Build meaning-based retrieval using the `Overview` field.

Use `text-embedding-3-small` embeddings plus FAISS for the local vector store.

### Document preparation

Each document should combine metadata with the overview for richer semantic matching:

```
"{Series_Title} ({Released_Year}) — {Genre} — Directed by {Director} — {Overview}"
```

Store `Series_Title`, `Released_Year`, `Genre`, `Director` as metadata fields in the index for post-retrieval filtering.

### Tasks

* generate embeddings for all ~1000 documents
* store the FAISS index locally in `vectorstore/`
* build retrieval function: `semantic_search(query: str, top_k: int, filters: dict) -> list[dict]`
* support metadata filtering after retrieval where needed

### This layer should support

| Assignment Q | Retrieval approach |
|---|---|
| Q7: Comedies with death/dead people | Semantic search on "death" + "dead people" concepts, filtered to comedy genre |
| Q8: Spielberg sci-fi plot summaries | Structured filter (director=Spielberg, genre contains sci-fi), then summarize retrieved overviews via LLM |
| Q9: Pre-1990 movies with police involvement | Semantic search for "police", "law enforcement", "cops", "detective" concepts, post-filtered to year < 1990 |

### Q8 note

Q8 is actually a hybrid: the retrieval is structured (director + genre filter), but the response requires LLM summarization of the plot overviews. Route this through the hybrid path.

### Q9 note

Q9 explicitly says "based on similarity search and not just the word 'police'." This is the strongest test of the semantic layer. The embedding model should naturally handle synonyms (law enforcement, detective, sheriff, etc.) but verify with spot checks.

Pushback: do not try to make semantic retrieval solve ranking and numeric filtering. That weakens correctness and makes the system harder to defend.

Exit condition:
Overview-based questions return relevant candidates with reasonable consistency. Q9 returns results that include non-literal police references.

## Workstream 5: query orchestration

This is the routing layer. It is the most failure-prone component — design it carefully.

### Routing mechanism: OpenAI function-calling

Use `gpt-4o-mini` with function-calling (tool use). Define each query capability as a tool:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_movie_info",
            "description": "Look up release year or basic info for a specific movie title",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "field": {"type": "string", "enum": ["release_year", "rating", "overview", "director"]}
                },
                "required": ["title", "field"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "top_movies_ranked",
            "description": "Get top-N movies ranked by a metric, with optional genre, year range, and threshold filters",
            "parameters": { ... }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_plot_search",
            "description": "Search movie plots by meaning for thematic questions",
            "parameters": { ... }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "director_aggregation",
            "description": "Find directors meeting specific criteria across their filmography",
            "parameters": { ... }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_clarification",
            "description": "Ask the user a clarifying question before executing the query",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["question"]
            }
        }
    }
]
```

### Why function-calling over alternatives

* **vs. classification prompt:** Function-calling gives structured, parseable output. A classification prompt requires output parsing and is more brittle.
* **vs. keyword heuristics:** Too rigid. Fails on paraphrased questions.
* Function-calling also extracts parameters (genre, year range, N) in one pass, avoiding a second extraction step.

### Routing paths

| Path | When used | Example |
|------|-----------|---------|
| Structured | Exact lookup, ranking, filtering, aggregation | "Top 5 movies of 2019 by meta score" |
| Semantic | Meaning-based plot retrieval | "Movies where police are involved" |
| Hybrid | Semantic retrieval + deterministic filters | "Pre-1990 police movies" or "Spielberg sci-fi summaries" |
| Clarification | Ambiguous intent needing follow-up | "Al Pacino movies over $50M" → lead vs. any role? |

### Clarification flow

For the Al Pacino nice-to-have:

1. Router detects actor query with ambiguous role scope
2. Calls `ask_clarification` tool with: "Are you looking for movies where Al Pacino is the lead actor, or any movie where Al Pacino appeared?"
3. User responds
4. Router re-invokes with the appropriate `role_scope` parameter

Store pending clarification state in Streamlit `session_state`.

Exit condition:
the system consistently chooses the correct path for all 9 core assignment prompts plus the Al Pacino clarification flow.

## Workstream 6: response construction

Once results are returned, build the response layer.

### Tasks

* format structured results into readable markdown tables
* summarize retrieved plot results when needed (use LLM for Q8 Spielberg summaries)
* keep wording grounded in returned data — no hallucinated facts
* handle no-results cases cleanly: "No movies matched your criteria."
* handle low-confidence semantic matches: "These results are based on plot similarity and may not be exact matches."
* support one-turn clarification resolution

### Reasoning and suggestions (assignment "nice-to-have")

The assignment says: "Provide reasoning and suggestions along with responses."

For each response, append:

* **Reasoning:** brief explanation of how the answer was derived (e.g., "Filtered 1000 movies to comedy genre, then ranked by IMDB rating within 2010–2020.")
* **Query path label:** "Answered via structured query" / "Answered via semantic retrieval" / "Answered via hybrid flow"

### Recommendation engine (nice-to-have #2)

After returning results, optionally suggest similar movies:

* For structured results: find movies with the closest `(IMDB_Rating, Meta_score)` Euclidean distance to the top result, excluding movies already in the result set
* This is a simple `pandas` operation — no LLM needed
* Append as: "You might also like: {title} ({year}) — IMDB {rating}, Meta {score}"

Keep this lightweight. It adds value with minimal implementation effort.

Exit condition:
responses are readable, grounded, include reasoning traces, and are consistent across query types.

## Workstream 7: Streamlit UI

Build the app only after the back-end logic is usable independently.

MVP UI should include:

* chat input with message history
* response area with markdown rendering
* tabular output for ranked/filter results (using `st.dataframe` or `st.table`)
* session state for follow-up clarification
* sample prompt sidebar with all 9 assignment questions as clickable examples
* query path label showing whether the answer came from structured, semantic, or hybrid flow
* simple error/empty-state handling

Do not overbuild the front end. The job of the UI is to make the flow understandable and demoable.

Exit condition:
a user can launch the app locally and interact with the system end to end.

## Workstream 8: MVP quality checks

This is the minimum quality layer inside MVP.

### Sanity checks — must-pass set

Cover the full difficulty spectrum, not just the easy cases:

| Check | Assignment Q | Type | Why included |
|-------|-------------|------|--------------|
| Release year lookup | Q1 | Structured / simple | Baseline correctness |
| Top-N by meta score | Q2 | Structured / ranking | Core ranking path |
| Genre + year-range filter | Q3 | Structured / compound | Multi-filter path |
| Director aggregation with "at least twice" | Q5 | Structured / hard | Most complex structured query — highest breakage risk |
| Semantic: comedies with death | Q7 | Semantic | Basic semantic retrieval |
| Semantic: police pre-1990 (similarity, not keyword) | Q9 | Semantic / hard | Most complex semantic query — tests non-literal matching |
| Hybrid: Spielberg sci-fi summaries | Q8 | Hybrid | Tests structured filter → LLM summarization pipeline |
| Clarification flow | Nice-to-have 1 | Clarification | Tests multi-turn interaction |

### Regression checks

Freeze expected results for the core supported prompts and rerun them after changes. Store expected outputs in `tests/expected_outputs/`.

### Edge-case checks

Test:

* movie title not found
* no matching results for valid filters
* missing gross or meta-score rows in filtered results
* ambiguous actor query triggering clarification
* misspelled title — graceful failure with suggestion ("Did you mean...?") or clean error
* empty genre after filtering
* year range that returns zero results

Pushback: this is not "full eval." It is MVP quality control. Useful, necessary, and small.

Exit condition:
core flows remain stable across small refactors. All 8 sanity checks pass.

## Recommended build order

### Step 1
Create repo structure and environment.

### Step 2
Load and clean dataset. Verify Gross parsing, Genre splitting, and cast normalization with spot checks.

### Step 3
Build and verify structured query functions. Test Q5 (director aggregation) explicitly — it is the hardest.

### Step 4
Build semantic retrieval and verify Overview-based questions. Test Q9 (police similarity) explicitly — it is the hardest.

### Step 5
Build orchestration logic with function-calling tools. Verify routing on all 9 assignment questions.

### Step 6
Build response formatter with reasoning traces and optional recommendations.

### Step 7
Integrate with Streamlit UI.

### Step 8
Run sanity, regression, and edge-case checks. Fix any routing or formatting issues.

## Suggested MVP file layout

```text
imdb_voice_agent/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── data/
│   └── imdb_top_1000.csv
├── vectorstore/
│   └── faiss_index/
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── structured_queries.py
│   ├── semantic_queries.py
│   ├── router.py
│   ├── llm_client.py
│   ├── response_builder.py
│   ├── recommender.py
│   └── utils.py
└── tests/
    ├── sanity_checks.py
    ├── regression_checks.py
    └── expected_outputs/
```

## MVP done criteria

The MVP is complete when:

* the app runs locally with `streamlit run app.py`
* the IMDB dataset is cleaned and queryable (including Gross nulls and multi-genre values)
* structured questions are answered deterministically via pandas
* semantic plot-based questions are answered through FAISS retrieval
* query routing uses OpenAI function-calling and consistently picks the correct path
* ambiguous questions trigger clarification before execution
* responses include reasoning traces and query-path labels
* the Streamlit app supports a usable conversational flow with message history
* all 8 sanity checks pass
* README documents: model choices, setup instructions, and how to run

## Post-MVP: voice support

The assignment title says "conversational voice agent." The MVP delivers the conversational agent; voice is a bolt-on layer.

### Path to voice

1. **Speech-to-text:** Use OpenAI Whisper API (`whisper-1`) or browser-native `webkitSpeechRecognition` via Streamlit's `audio_input` component
2. **Text-to-speech:** Use OpenAI TTS API (`tts-1` with `alloy` voice) to read responses aloud
3. **Integration point:** Add a mic button to the Streamlit UI. Audio → Whisper → text query → existing pipeline → text response → TTS → audio playback

### Effort estimate

* ~2–3 hours if using OpenAI's Whisper + TTS APIs
* Streamlit has `st.audio` for playback and community components for mic input
* No architectural changes to the core pipeline — voice is a thin I/O wrapper

### Why defer

* voice adds API cost and latency without improving answer quality
* the assignment lists it in the problem statement but all test questions are text-based
* better to nail correctness first, then layer on voice if time permits

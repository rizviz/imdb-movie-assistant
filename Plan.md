Apr-12-2026 , 10:49 AM CST

## MVP dev plan

### MVP objective

Build a local Streamlit-based conversational movie assistant that uses the IMDB dataset to answer structured and semantic questions reliably.

The MVP ends when the app can run locally, answer the core assignment questions, and support a basic conversational flow with clarification where needed.

### Core MVP scope

The MVP should include only these core capabilities:

1. dataset ingestion and preprocessing
2. structured query engine
3. semantic retrieval over plot summaries
4. query orchestration
5. response construction
6. Streamlit UI
7. basic quality checks

Voice, advanced recommendations, Docker packaging, and extra polish should stay outside the MVP unless the core is already stable.

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

Tasks:

* load CSV from the zip
* inspect column quality and nulls
* normalize `Released_Year`
* convert `IMDB_Rating`, `Meta_score`, `No_of_votes`, and `Gross` into numeric form
* standardize `Genre`
* combine or normalize cast fields from `Star1` to `Star4`
* clean `Overview`
* create helper columns needed for filtering and retrieval

Output:

* one cleaned dataframe for structured querying
* one text-ready representation for semantic indexing

Exit condition:
a single clean dataframe is available and can support exact filters and rankings without ad hoc fixes later.

## Workstream 3: structured query engine

Build deterministic functions for exact questions.

This layer should handle:

* title lookups
* release year lookups
* top-N ranking by score
* year range filters
* genre filters
* rating and meta-score thresholds
* vote-count and gross filters
* director aggregations
* actor-based filtering

Representative functions:

* movie release year by title
* top movies by year and meta score
* top comedies in a year range by IMDB rating
* horror movies above threshold values
* directors with multiple movies above a gross threshold
* high-vote, low-gross movie lists

This layer should return raw structured results, not LLM prose.

Exit condition:
all exact and ranking-style assignment questions can be answered directly from pandas.

## Workstream 4: semantic retrieval layer

Build meaning-based retrieval using the `Overview` field.

Use embeddings plus a local vector store such as FAISS or Chroma.

Tasks:

* decide document text format, likely title + genre + director + overview
* generate embeddings
* store the index locally
* build retrieval functions
* support metadata filtering after retrieval where needed

This layer should support:

* death/dead people in comedy plots
* police involvement before 1990 based on meaning, not just keyword match
* Spielberg sci-fi plot retrieval before summarization

Pushback: do not try to make semantic retrieval solve ranking and numeric filtering. That weakens correctness and makes the system harder to defend.

Exit condition:
Overview-based questions return relevant candidates with reasonable consistency.

## Workstream 5: query orchestration

This is the routing layer.

It should decide whether a user question belongs to:

* structured path
* semantic path
* hybrid path
* clarification path

Structured path:
exact lookup, ranking, filtering, aggregation

Semantic path:
meaning-based plot retrieval

Hybrid path:
semantic retrieval plus deterministic filters, such as pre-1990 police-related plots

Clarification path:
questions that need a follow-up before execution, such as the Al Pacino case

Keep the orchestration logic narrow and explicit. The LLM is not the source of truth here. It is only helping route and phrase responses.

Exit condition:
the system consistently chooses the correct path for the core assignment prompts.

## Workstream 6: response construction

Once results are returned, build the response layer.

Tasks:

* format structured results into readable tables or short summaries
* summarize retrieved plot results when needed
* keep wording grounded in returned data
* handle no-results cases cleanly
* handle low-confidence semantic matches carefully
* support one-turn clarification resolution

This layer should also attach a simple trace such as:

* Answered via structured query
* Answered via semantic retrieval
* Answered via hybrid flow

That is useful both for debugging and for explaining the system.

Exit condition:
responses are readable, grounded, and consistent across query types.

## Workstream 7: Streamlit UI

Build the app only after the back-end logic is usable independently.

MVP UI should include:

* chat input
* response area
* tabular output for ranked/filter results
* session state for follow-up clarification
* sample prompt section or sidebar
* optional label showing the query path used

Do not overbuild the front end. The job of the UI is to make the flow understandable and demoable.

Exit condition:
a user can launch the app locally and interact with the system end to end.

## Workstream 8: MVP quality checks

This is the minimum quality layer inside MVP.

### Sanity checks

Create a must-pass set covering:

* release year lookup
* top-N ranking by meta score
* genre + year-range filtering
* one semantic Overview query
* one ambiguity/clarification flow

### Regression checks

Freeze expected results for the core supported prompts and rerun them after changes.

### Edge-case checks

Test:

* movie title not found
* no matching results
* missing gross or meta-score rows
* ambiguous actor query
* misspelled title with graceful fallback or failure

Pushback: this is not “full eval.” It is MVP quality control. Useful, necessary, and small.

Exit condition:
core flows remain stable across small refactors.

## Recommended build order

### Step 1

Create repo structure and environment.

### Step 2

Load and clean dataset.

### Step 3

Build and verify structured query functions.

### Step 4

Build semantic retrieval and verify Overview-based questions.

### Step 5

Build orchestration logic.

### Step 6

Build response formatter.

### Step 7

Integrate with Streamlit UI.

### Step 8

Add sanity, regression, and edge-case checks.

## Suggested MVP file layout

```text
imdb_voice_agent/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── data/
├── vectorstore/
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── structured_queries.py
│   ├── semantic_queries.py
│   ├── router.py
│   ├── llm_client.py
│   ├── response_builder.py
│   └── utils.py
└── tests/
    ├── sanity_checks.py
    └── regression_checks.py
```

## MVP done criteria

The MVP is complete when:

* the app runs locally
* the IMDB dataset is cleaned and queryable
* structured questions are answered deterministically
* semantic plot-based questions are answered through retrieval
* ambiguous questions trigger clarification
* the Streamlit app supports a usable conversational flow
* sanity and regression checks pass


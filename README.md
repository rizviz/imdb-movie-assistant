# 🎬 IMDB Conversational Movie Assistant

A Gen AI powered conversational agent that uses the IMDB Top 1000 dataset to answer structured and semantic questions about movies.

## Features

- **Structured Queries**: Deterministic pandas-based lookups for exact answers (title lookup, rankings, filtering, aggregations)
- **Semantic Search**: FAISS-based embedding similarity search over movie plot summaries
- **Hybrid Queries**: Combines structured filtering with LLM summarization
- **Smart Routing**: OpenAI function-calling for automatic query classification
- **Clarification Flow**: Asks follow-up questions when user intent is ambiguous
- **Recommendations**: Suggests similar movies based on rating proximity
- **Reasoning Traces**: Shows how each answer was derived

## Models Used

| Role | Model | Rationale |
|------|-------|-----------|
| Routing / Orchestration | `gpt-4o-mini` | Fast, cheap, strong at function-calling |
| Summarization / Response | `gpt-4o-mini` | Sufficient for narrative answers over retrieved data |
| Embeddings | `text-embedding-3-small` | 1536-dim, low cost, good semantic quality |

## Setup

### Prerequisites
- Python 3.9+
- OpenAI API key

### Installation

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key

# 4. Run the app
streamlit run app.py
```

## Project Structure

```
├── app.py                    # Streamlit UI
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── README.md                 # This file
├── data/
│   └── imdb_top_1000.csv     # IMDB dataset
├── vectorstore/
│   └── faiss_index/          # Persisted FAISS index
├── src/
│   ├── data_loader.py        # CSV loading
│   ├── preprocess.py         # Data cleaning & normalization
│   ├── structured_queries.py # Deterministic query functions
│   ├── semantic_queries.py   # FAISS semantic search
│   ├── router.py             # Query orchestration (function-calling)
│   ├── llm_client.py         # OpenAI API wrapper
│   ├── response_builder.py   # Response formatting
│   ├── recommender.py        # Movie recommendations
│   └── utils.py              # Shared utilities
├── tests/
│   ├── sanity_checks.py      # Data quality & query correctness tests
│   └── eval_runner.py        # End-to-end evaluation runner
└── memory.md                 # Development log
```

## Test Questions Supported

1. When did The Matrix release?
2. What are the top 5 movies of 2019 by meta score?
3. Top 7 comedy movies between 2010-2020 by IMDB rating?
4. Top horror movies with a meta score above 85 and IMDB rating above 8
5. Top directors and their highest grossing movies with gross earnings of greater than 500M at least twice
6. Top 10 movies with over 1M votes but lower gross earnings
7. List of movies from the comedy genre where there is death or dead people involved
8. Summarize the movie plots of Steven Spielberg's top-rated sci-fi movies
9. List of movies before 1990 that have involvement of police in the plot

### Nice-to-haves
- Al Pacino movies with gross > $50M and IMDB ≥ 8 (with lead vs. any role clarification)
- Movie recommendations with similar ratings

## Running Tests

```bash
# Sanity checks (no API calls needed)
source .venv/bin/activate
pytest tests/sanity_checks.py -v

# End-to-end evaluation (requires API key)
python tests/eval_runner.py
```

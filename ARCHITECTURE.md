# Model Selection & Architecture Decisions

## Model Stack

| Role | Model | Why |
|------|-------|-----|
| Routing / orchestration | `gpt-4o-mini` | Best function-calling reliability at lowest cost. Handles intent classification + parameter extraction in a single API call |
| Summarization | `gpt-4o-mini` | Sufficient quality for summarizing retrieved plot data. Upgradeable to `gpt-4o` for richer narratives |
| Embeddings | `text-embedding-3-small` | 1536-dim, $0.02/1M tokens. Strong semantic quality for plot-level retrieval at this dataset scale |
| Speech-to-text | `whisper-1` | Reliable transcription, same SDK, single API call |
| Text-to-speech | `tts-1` | Fast and cheap for MVP voice playback |
| Vector store | FAISS (IndexFlatIP) | Local, zero infrastructure, persists to disk. Assignment requires sharing the vectorstore file |

## Selection Criteria

Models were selected based on these criteria, in priority order:

1. **Task fit** — Does the model reliably perform the specific task?
2. **Latency** — Can it respond within conversational UX expectations (< 2s routing, < 5s summarization)?
3. **Cost** — Cheap enough for development iteration and demo ($0.01 for all 10 eval questions)
4. **Simplicity** — No fine-tuning, no self-hosted models, no GPU requirements

## Architecture Decisions

### Why function-calling instead of prompt-based routing?

Prompt-based classification (e.g., "Is this structured or semantic?") is unreliable — the LLM might hallucinate categories. Function-calling is deterministic: the LLM outputs structured JSON with the function name and typed parameters. Intent classification and parameter extraction happen in one call.

### Why separate structured and semantic paths?

Structured queries (Q1–Q6) are deterministic and don't need an LLM to answer. Running them through pandas is faster (< 1ms), cheaper (zero API cost), and guaranteed correct. The LLM is only used where it adds value: routing decisions, semantic understanding, and summarization.

### Why FAISS instead of a hosted vector DB?

For 1,000 documents, exact nearest neighbor search takes < 1ms. A hosted DB (Pinecone, Weaviate) would add network latency, API keys, and infrastructure complexity for zero performance gain. FAISS persists to two files (`index.faiss` + `metadata.json`) that ship with the repo.

### Why preserve NaN for Gross instead of filling with zero?

16.9% of movies have missing gross data. Filling with zero would falsely rank movies with unknown gross as "lowest grossing" — breaking Q6 and any gross-based ranking. A `has_gross` boolean flag lets queries explicitly include/exclude.

### Why post-retrieval filtering instead of pre-filtering embeddings?

Embeddings capture semantic meaning but can't enforce hard categorical constraints. A movie about "funny death scenes" might have high cosine similarity to a "comedy + death" query but actually be classified as Drama. Post-retrieval filtering guarantees exact genre/year/director constraints after semantic ranking.

### Why a hybrid path for Q8?

"Summarize Spielberg's sci-fi plots" requires both:
- **Exact filtering** (Director = 'Steven Spielberg' AND Genre contains 'Sci-Fi') — can't risk semantic drift
- **Generative summarization** (plot overviews → coherent summary) — can't do with pandas

Neither structured nor semantic alone can answer this. The hybrid path chains them.

### Grounding strategy

The LLM never generates movie facts from its training data. It serves two roles:
1. **Router** — selects which tool to call (function-calling)
2. **Summarizer** — condenses retrieved plot text (only for Q8)

All factual data comes from the IMDB dataset via pandas or FAISS. This is a structural guardrail — even prompt injection can't cause hallucinated movie data.

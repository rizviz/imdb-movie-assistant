"""
LLM Client — handles all OpenAI API interactions.

Provides: chat completions, function-calling, and embeddings.
"""
import openai
from src.utils import get_openai_api_key, get_llm_model, get_embedding_model


_client = None


def get_client() -> openai.OpenAI:
    """Get or create the OpenAI client singleton."""
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=get_openai_api_key())
    return _client


def chat_completion(
    messages: list,
    model: str = None,
    tools: list = None,
    tool_choice: str = "auto",
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> dict:
    """
    Make a chat completion call, optionally with function-calling tools.
    Returns the full response message.
    """
    client = get_client()
    model = model or get_llm_model()

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message


def generate_embedding(text: str, model: str = None) -> list[float]:
    """Generate an embedding vector for a single text string."""
    client = get_client()
    model = model or get_embedding_model()
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def generate_embeddings_batch(texts: list[str], model: str = None, batch_size: int = 100) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    client = get_client()
    model = model or get_embedding_model()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        if i + batch_size < len(texts):
            print(f"  [Embeddings] Processed {i + batch_size}/{len(texts)}")

    return all_embeddings


def summarize_plots(plots: list[dict], context: str = "") -> str:
    """
    Use LLM to summarize movie plot overviews.
    Used for Q8 (Spielberg sci-fi summaries).
    """
    plot_texts = "\n\n".join(
        f"**{p['title']}** ({p.get('year', 'N/A')}): {p['overview']}"
        for p in plots
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful movie assistant. Summarize the following movie plots "
                "concisely. Highlight key themes and what makes each movie notable. "
                "Keep your summary informative but brief."
            ),
        },
        {
            "role": "user",
            "content": f"{context}\n\nHere are the movie plots to summarize:\n\n{plot_texts}",
        },
    ]

    response = chat_completion(messages, temperature=0.5, max_tokens=1500)
    return response.content

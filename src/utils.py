"""
Utility functions shared across the IMDB movie assistant.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def get_env(key: str, default: str = None) -> str:
    """Retrieve an environment variable or raise if missing and no default."""
    value = os.getenv(key, default)
    if value is None:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value


def get_openai_api_key() -> str:
    return get_env("OPENAI_API_KEY")


def get_llm_model() -> str:
    return get_env("LLM_MODEL", "gpt-4o-mini")


def get_llm_model_fallback() -> str:
    return get_env("LLM_MODEL_FALLBACK", "gpt-4o")


def get_embedding_model() -> str:
    return get_env("EMBEDDING_MODEL", "text-embedding-3-small")


def get_faiss_index_path() -> str:
    return get_env("FAISS_INDEX_PATH", "vectorstore/faiss_index")


def format_currency(value) -> str:
    """Format a numeric value as USD currency string."""
    if value is None or (hasattr(value, '__class__') and str(value) == 'nan'):
        return "N/A"
    try:
        import math
        if math.isnan(float(value)):
            return "N/A"
    except (ValueError, TypeError):
        return "N/A"
    return f"${value:,.0f}"


def format_number(value) -> str:
    """Format a numeric value with commas."""
    if value is None:
        return "N/A"
    try:
        import math
        if math.isnan(float(value)):
            return "N/A"
    except (ValueError, TypeError):
        return "N/A"
    return f"{value:,.0f}"

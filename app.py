"""
IMDB Conversational Movie Assistant — Streamlit App

A Gen AI powered conversational agent that uses the IMDB Top 1000 dataset
to answer structured and semantic questions about movies.
"""
import streamlit as st
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import get_clean_dataframe
from src.semantic_queries import get_index_and_metadata
from src.router import route_query


# ─────────────────────────────
# Page Config
# ─────────────────────────────
st.set_page_config(
    page_title="🎬 IMDB Movie Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────
# Sample Prompts (Assignment Qs)
# ─────────────────────────────
SAMPLE_PROMPTS = [
    "When did The Matrix release?",
    "What are the top 5 movies of 2019 by meta score?",
    "Top 7 comedy movies between 2010-2020 by IMDB rating?",
    "Top horror movies with a meta score above 85 and IMDB rating above 8",
    "Top directors and their highest grossing movies with gross earnings of greater than 500M at least twice.",
    "Top 10 movies with over 1M votes but lower gross earnings.",
    "List of movies from the comedy genre where there is death or dead people involved.",
    "Summarize the movie plots of Steven Spielberg's top-rated sci-fi movies.",
    "List of movies before 1990 that have involvement of police in the plot.",
    "List of Al Pacino movies that have grossed over $50M and IMDB rating of 8 or higher.",
]


# ─────────────────────────────
# Initialize Session State
# ─────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "index_built" not in st.session_state:
    st.session_state.index_built = False
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None


# ─────────────────────────────
# Load Data
# ─────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    """Load and preprocess the IMDB dataset."""
    return get_clean_dataframe()


def init_data():
    """Initialize the dataset and FAISS index."""
    if st.session_state.df is None:
        with st.spinner("🔄 Loading IMDB dataset..."):
            st.session_state.df = load_data()

    if not st.session_state.index_built:
        with st.spinner("🧠 Building semantic search index (this may take a minute on first run)..."):
            get_index_and_metadata(st.session_state.df)
            st.session_state.index_built = True


# ─────────────────────────────
# Sidebar
# ─────────────────────────────
with st.sidebar:
    st.title("🎬 Movie Assistant")
    st.markdown("---")

    st.markdown("### 📝 Sample Questions")
    st.markdown("Click any question below to try it:")

    for i, prompt in enumerate(SAMPLE_PROMPTS):
        label = f"Q{i+1}" if i < 9 else "Bonus"
        if st.button(f"{label}: {prompt[:50]}...", key=f"sample_{i}", use_container_width=True):
            st.session_state.sample_prompt = prompt

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This assistant uses **OpenAI GPT-4o-mini** for query routing "
        "and **FAISS** for semantic search over movie plot summaries."
    )
    st.markdown(
        "**Models used:**\n"
        "- Routing: `gpt-4o-mini`\n"
        "- Embeddings: `text-embedding-3-small`\n"
        "- Summarization: `gpt-4o-mini`"
    )

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_clarification = None
        st.rerun()


# ─────────────────────────────
# Main App
# ─────────────────────────────
st.title("🎬 IMDB Conversational Movie Assistant")
st.caption("Ask me anything about the top 1000 IMDB movies!")

# Initialize data
init_data()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle sample prompt from sidebar
if "sample_prompt" in st.session_state and st.session_state.sample_prompt:
    prompt = st.session_state.sample_prompt
    st.session_state.sample_prompt = None

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Build conversation history for context
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]  # exclude current
                ]

                response_text, query_type, raw_result = route_query(
                    prompt,
                    st.session_state.df,
                    conversation_history=history,
                    pending_clarification=st.session_state.pending_clarification,
                )

                # Handle clarification state
                if query_type == "clarification" and raw_result.get("pending_function"):
                    st.session_state.pending_clarification = raw_result
                else:
                    st.session_state.pending_clarification = None

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

    st.rerun()

# Chat input
if prompt := st.chat_input("Ask me about movies..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

                response_text, query_type, raw_result = route_query(
                    prompt,
                    st.session_state.df,
                    conversation_history=history,
                    pending_clarification=st.session_state.pending_clarification,
                )

                if query_type == "clarification" and raw_result.get("pending_function"):
                    st.session_state.pending_clarification = raw_result
                else:
                    st.session_state.pending_clarification = None

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

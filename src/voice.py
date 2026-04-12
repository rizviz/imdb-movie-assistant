"""
Voice Module — minimum viable voice input/output.

Speech-to-text via OpenAI Whisper API (whisper-1).
Text-to-speech via OpenAI TTS API (tts-1, voice alloy).
"""
import io
import re
from src.llm_client import get_client


def transcribe_audio(audio_bytes: bytes, filename: str = "recording.wav") -> str:
    """
    Transcribe audio bytes to text using OpenAI Whisper API.
    
    Returns the transcribed text, or an empty string if transcription fails.
    """
    client = get_client()

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename

    try:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )
        text = response.strip() if isinstance(response, str) else str(response).strip()
        return text
    except Exception as e:
        print(f"[Voice] Transcription error: {e}")
        return ""


def text_to_speech(text: str, voice: str = "alloy") -> bytes:
    """
    Convert text to speech using OpenAI TTS API.
    
    Returns MP3 audio bytes.
    """
    client = get_client()

    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
        )
        return response.content
    except Exception as e:
        print(f"[Voice] TTS error: {e}")
        return b""


def generate_spoken_summary(response_text: str, query_type: str) -> str:
    """
    Extract a short, TTS-friendly summary from a full response.
    
    Long responses (tables, lists) get condensed to 2-3 sentences.
    Short responses (lookups, clarifications) are read as-is.
    """
    # Strip markdown formatting for cleaner speech
    clean = _strip_markdown(response_text)

    # Clarification — read as-is
    if query_type == "clarification":
        # Just get the question part
        if "Clarification needed:" in clean:
            parts = clean.split("Clarification needed:")
            return "I need a quick clarification. " + parts[1].split("Options:")[0].strip()
        return clean[:300]

    # Lookup — short, read as-is
    if query_type == "lookup":
        # Get just the first line (the answer)
        first_line = clean.split("\n")[0].strip()
        return first_line[:300]

    # Structured / semantic / hybrid — summarize
    # Try to extract key info
    lines = clean.split("\n")

    # Find "Found N matching movies" or similar header
    header = ""
    for line in lines[:5]:
        if "Found" in line and "movie" in line.lower():
            header = line.strip()
            break

    # Extract first few movie titles
    titles = []
    for line in lines:
        # Look for numbered entries like "1. Title" or "| 1 | Title"
        match = re.match(r'(?:\d+\.\s+|^\|\s*\d+\s*\|\s*)(.+?)(?:\s*\||\s*\()', line)
        if match:
            title = match.group(1).strip().strip('*').strip()
            if title and len(title) > 2:
                titles.append(title)
        if len(titles) >= 3:
            break

    if header and titles:
        title_list = ", ".join(titles[:3])
        remaining = len(titles) - 3
        summary = f"{header}. The top results include {title_list}"
        if remaining > 0:
            summary += f", and {remaining} more"
        summary += "."
        return summary

    if header:
        return header

    # Fallback: first 2 sentences
    sentences = re.split(r'[.!?]+', clean)
    summary = ". ".join(s.strip() for s in sentences[:2] if s.strip())
    return (summary + ".") if summary else "Here are the results."


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting for cleaner TTS output."""
    # Remove markdown bold/italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # Remove emoji
    text = re.sub(r'[🎬🔍📋🤔💡❌⚠️ℹ️❓🔄]', '', text)
    # Remove markdown table formatting
    text = re.sub(r'\|[-\s]+\|', '', text)
    # Remove horizontal rules
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    # Remove markdown links
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

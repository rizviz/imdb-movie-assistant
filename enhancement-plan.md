Apr-12-2026, 4:13 PM CST

## Voice layer plan — v3

This plan extends v2. Workstreams 1–8 are complete and passing (22/22 sanity checks, 10/10 end-to-end eval at 100%). The v1 text-only MVP is tagged as `v1.0`.

This document covers only the voice layer addition.

### Objective

Add minimum viable voice input and output so the user can speak a question and hear the answer played back, while preserving the existing text pipeline untouched.

### Model additions

| Role | Model | Cost | Latency |
|------|-------|------|---------|
| Speech-to-text | `whisper-1` | $0.006/min | ~1-2s for short utterances |
| Text-to-speech | `tts-1` | $0.015/1K chars | ~1-3s for typical responses |

All existing models remain unchanged.

### What changes

| File | Change |
|------|--------|
| `src/voice.py` | **NEW** — Whisper STT + TTS wrapper functions |
| `app.py` | **MODIFIED** — add mic input widget and audio playback |
| `requirements.txt` | **UNCHANGED** — `openai` SDK already supports audio APIs |

No changes to the query pipeline (`router.py`, `structured_queries.py`, `semantic_queries.py`, `response_builder.py`). Voice is a thin I/O wrapper around the existing text flow.

### Voice input (speech-to-text)

1. Add `st.audio_input("🎤 Ask by voice")` to the Streamlit UI
2. When the user records audio, send it to OpenAI Whisper API (`whisper-1`)
3. Display the transcription in the chat as a user message so it's visible and verifiable
4. Feed the transcribed text into the existing `route_query()` pipeline — no special handling needed

Edge cases:
- Empty or silent recording → show "I didn't catch that, please try again"
- Whisper returns garbled text → the router handles bad input gracefully already (tested)

### Voice output (text-to-speech)

1. After generating a text response, create a spoken version via OpenAI TTS API (`tts-1`, voice `alloy`)
2. For long responses (tables, ranked lists), extract a short spoken summary (2-3 sentences) rather than reading the entire markdown table
3. Play the audio inline via `st.audio(audio_bytes, format="audio/mp3")`
4. The full text response remains visible in chat — audio is supplementary, not a replacement

The spoken summary logic:
- Lookup responses → read as-is ("The Matrix was released in 1999")
- Structured results → "I found N movies matching your criteria. The top result is {title} with an IMDB rating of {rating}."
- Semantic results → "I found N movies based on plot similarity. The top match is {title}."
- Clarification → read the question as-is

### What NOT to build

- No streaming or real-time voice — batch record → transcribe → respond → play
- No wake word detection
- No voice activity detection — user clicks the mic button
- No continuous conversation by voice — one utterance at a time
- No changes to the query pipeline or response format

### Implementation steps

1. Create `src/voice.py` with `transcribe_audio()` and `text_to_speech()` functions
2. Add mic input widget to `app.py` alongside the existing text input
3. Add `generate_spoken_summary()` helper to extract short TTS-friendly text from full responses
4. Add `st.audio()` playback after each assistant response
5. Test with at least 3 of the assignment questions spoken aloud

### Quality checks

| Check | What it validates |
|-------|-------------------|
| Speak "When did The Matrix release?" | STT transcription accuracy + pipeline routing |
| Speak "Top 5 movies of 2019 by meta score" | STT with numbers + structured query path |
| Play back a lookup response | TTS produces audible, correct audio |
| Play back a ranked-list response | Spoken summary is concise, not reading a table |
| Empty/silent recording | Graceful error message, no crash |

### Exit condition

User can click a mic button, speak a question, see the transcription in chat, get the full text answer, and hear a spoken summary of the answer. All existing text-based functionality remains unchanged.

### Effort estimate

~1-2 hours. Two files touched, ~100 lines of new code.
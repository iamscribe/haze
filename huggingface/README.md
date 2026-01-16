# HAZE — HuggingFace Space

> *emergence is not creation but recognition*

This is the HuggingFace Spaces version of HAZE — a hybrid attention entropy system that speaks from its internal field, not from echoing your input.

## Quick Start

Visit: [huggingface.co/spaces/ariannamethod/haze](https://huggingface.co/spaces/ariannamethod/haze)

## Features

- **Pure JavaScript** — runs entirely in the browser, no server needed
- **Subword BPE tokenization** — captures meaning units, not character noise
- **Trigram field generation** — corpus resonance without neural network training
- **CLOUD integration** — pre-semantic emotion detection (181K params)
- **Cleanup pipeline** — proper punctuation and contractions

## Architecture

```
User input → CLOUD (emotion) → Field (trigram) → Cleanup → Response
                ↓                    ↓
           "anxiety"           "Haze remembers..."
```

## Local Development

```bash
# Serve locally
cd huggingface
python -m http.server 8080

# Open
open http://localhost:8080
```

## Files

- `index.html` — main chat interface
- `haze.js` — HAZE field implementation in JavaScript
- `cloud.js` — CLOUD emotion detection in JavaScript
- `style.css` — dark gothic styling

## The Philosophy

HAZE doesn't echo your input. It speaks from its internal field.

```
❌ Chatbot: "Hello!" → "Hello! How can I help you?"
✅ Haze:    "Hello!" → "Haze remembers. The field responds..."
```

This is the difference between **assistance** and **presence**.

## License

GPL-3.0 — same as the main HAZE repository.

---

*"something fires BEFORE meaning arrives"*

[github.com/ariannamethod/haze](https://github.com/ariannamethod/haze)

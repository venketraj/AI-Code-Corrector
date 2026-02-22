# Code Rewriter with Guidelines

A Streamlit web app that refactors Python or C code using AI — either a local Ollama model or the Mistral Codestral cloud API — following user-defined guidelines. All runs are saved to Supabase for history and replay.

---

## Features

- **Guideline-driven refactoring** — Paste or upload a set of rules (e.g. "add comments", "optimize loops", "use const where possible") and the model applies every rule to the provided code.
- **Two language targets** — Supports **Python** (`.py`) and **C** (`.c`) source files.
- **Two AI providers**
  - **Ollama (local)** — Runs Mistral 7B Instruct (or any Ollama-compatible model) on your own machine. No API key required.
  - **Mistral Codestral (cloud)** — Uses the Codestral API for faster, higher-quality results. Requires a Mistral API key.
- **File upload or paste** — Guidelines and source code can be uploaded as files or typed directly into text areas.
- **Structured output** — The model returns a compact JSON object containing:
  - `summary` — A markdown bulleted list of every change made, with the guideline name, affected function, and line numbers.
  - `code` — The fully rewritten, functional source code.
- **Download rewritten code** — One-click download of the refactored file in the correct extension.
- **Supabase history** — Every generation is automatically saved (guidelines, input code, output code, model used, runtime). The sidebar lets you browse the last 50 runs and reload any of them.

---

## Project Structure

```
code-corrector/
├── code_corrector.py   # Main Streamlit application
├── README.md
└── .streamlit/
    └── secrets.toml    # API keys and Supabase credentials (not committed)
```

---

## Requirements

- Python 3.9+
- [Streamlit](https://streamlit.io/)
- [Supabase Python client](https://github.com/supabase-community/supabase-py)
- `requests`

Install dependencies:

```bash
pip install streamlit supabase requests
```

---

## Configuration

Create `.streamlit/secrets.toml` in the project root:

```toml
[supabase]
url = "https://your-project.supabase.co"
anon_key = "your-supabase-anon-key"

[codestral]
api_key = "your-mistral-api-key"
base_url = "https://codestral.mistral.ai"
```

### Optional environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Base URL of your Ollama server |
| `OLLAMA_MODEL` | `mistral` | Ollama model name |
| `CODESTRAL_MODEL` | `codestral-latest` | Mistral Codestral model name |
| `SUPABASE_TABLE` | `code_transactions` | Supabase table used for history |

---

## Supabase Table Schema

Create a table named `code_transactions` (or your custom name) with these columns:

| Column | Type |
|---|---|
| `id` | `uuid` (primary key, default `gen_random_uuid()`) |
| `created_at` | `timestamptz` (default `now()`) |
| `guidelines` | `text` |
| `input_code` | `text` |
| `output_code` | `text` |
| `model` | `text` |
| `runtime_s` | `int4` |
| `summary` | `text` |

---

## Running the App

```bash
streamlit run code_corrector.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Select a language** — Choose Python or C from the dropdown.
2. **Choose a provider** — Pick Ollama (local) or Mistral Codestral (cloud).
   - For Ollama, confirm the server URL and model name.
   - For Codestral, ensure your API key is set in `secrets.toml`.
3. **Provide guidelines** — Upload a `.txt`/`.md` file or paste rules into the text area.
4. **Provide source code** — Upload a source file or paste code into the text area.
5. **Click "Generate changes and code"** — The model rewrites the code and returns:
   - A summary of every change, organized by guideline and line number.
   - A download button for the refactored file.
6. **Browse history** — Use the sidebar to view past runs, reload inputs/outputs, or compare results across different models.

---

## Notes

- Temperature is fixed at `0.4` for deterministic, consistent refactoring output.
- If the model returns malformed JSON, the app falls back to a heuristic JSON extractor before displaying raw output.
- Supabase is optional; the app degrades gracefully if credentials are missing or the connection fails.
- The Ollama provider has a 600-second request timeout to accommodate large models running locally.

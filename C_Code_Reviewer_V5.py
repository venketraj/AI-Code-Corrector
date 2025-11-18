import os
import time
import json
from typing import Optional, Dict, Any, List

import requests
import streamlit as st

# Optional: supabase client
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# -------------------------
# App config and constants
# -------------------------
st.set_page_config(page_title="Code Rewriter with Guidelines", page_icon="\U0001F9E9")
st.title("Code Rewriter with Guidelines " + "\U0001F9E9")


# Defaults and env
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
DEFAULT_CODESTRAL_MODEL = os.getenv("CODESTRAL_MODEL", "codestral-latest")
DEFAULT_SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "code_transactions")

SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["anon_key"]
MISTRAL_API_KEY = st.secrets["codestral"]["api_key"]
MISTRAL_API_URL = st.secrets["codestral"]["base_url"]

# -------------------------
# Styles
# -------------------------
st.markdown(
    """
    <style>
    .small-caption { font-size: 0.85rem; color: #666; }
    .ok { color: #0a0; }
    .warn { color: #aa0; }
    .err { color: #a00; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Supabase setup
# -------------------------
supabase: Optional["Client"] = None
if SUPABASE_URL and SUPABASE_KEY and create_client:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        supa_status = "Supabase connected"
        supa_class = "ok"
    except Exception as e:
        supabase = None
        supa_status = f"Supabase error: {e}"
        supa_class = "err"
else:
    supa_status = "Supabase disabled or client missing"
    supa_class = "warn"

st.markdown(f'<div class="small-caption {supa_class}">{supa_status}</div>', unsafe_allow_html=True)

# -------------------------
# Programming Laungauge Selection
# -------------------------
# Laungauge selection UI
# -------------------------
st.markdown("Provide guidelines and code. Choose programming laungauge and local or cloud model provider, then generate.")
Laungauge = st.selectbox("Laungauge", ["C", "Python"])


# -------------------------
# Providers
# -------------------------
PROVIDER_OLLAMA = "Ollama (local Mistral 7B Instruct)"
PROVIDER_CODESTRAL = "Mistral Codestral (cloud)"

# -------------------------
# Provider selection UI
# -------------------------
st.markdown("Provide guidelines and C code. Choose local or cloud model provider, then generate.")
provider = st.selectbox("Provider", [PROVIDER_OLLAMA, PROVIDER_CODESTRAL])

if provider == PROVIDER_OLLAMA:
    ollama_url = st.text_input("Ollama URL", DEFAULT_OLLAMA_URL)
    ollama_model = st.text_input("Ollama model", DEFAULT_OLLAMA_MODEL)
    st.caption(f"Using local model via Ollama at {ollama_url}")
else:
    codestral_model = st.text_input("Codestral model", DEFAULT_CODESTRAL_MODEL)
    api_key_masked = "set" if bool(MISTRAL_API_KEY) else "missing"
    st.caption(f"MISTRAL_API_KEY is {api_key_masked}. Configure in environment.")

# -------------------------
# Controls
# -------------------------
temperature = 0.4
max_tokens = 32000

# Session defaults for history load
for key, default in [
    ("loaded_guidelines", ""),
    ("loaded_input_code", ""),
    ("loaded_output_code", ""),
    ("loaded_model", ""),
    ("loaded_runtime_s", 0),
    ("loaded_summary", ""),
    ("just_generated", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------
# Sidebar: History
# -------------------------
def load_recent_transactions(limit: int = 25) -> List[Dict[str, Any]]:
    if not supabase:
        return []
    res = supabase.table(DEFAULT_SUPABASE_TABLE).select("*").order("created_at", desc=True).limit(limit).execute()
    return res.data or []

def load_transaction_by_id(record_id: str) -> Optional[Dict[str, Any]]:
    if not supabase:
        return None
    res = supabase.table(DEFAULT_SUPABASE_TABLE).select("*").eq("id", record_id).limit(1).execute()
    rows = res.data or []
    return rows[0] if rows else None

def save_transaction(guidelines: str, input_code: str, output_code: str, model: str, runtime_s: int, summary: str) -> Optional[str]:
    if not supabase:
        return None
    payload = {
        "guidelines": guidelines,
        "input_code": input_code,
        "output_code": output_code,
        "model": model,
        "runtime_s": runtime_s,
        "summary": summary
    }
    res = supabase.table(DEFAULT_SUPABASE_TABLE).insert(payload).execute()
    if hasattr(res, "data") and res.data:
        return res.data[0].get("id")
    return None

with st.sidebar:
    st.subheader("History")
    if supabase:
        txns = load_recent_transactions(limit=50)
        if not txns:
            st.caption("No records yet.")
        else:
            labels = [
                f"{t.get('created_at','')} 路 {t.get('model','')} 路 {str(t.get('id'))[:8]}"
                for t in txns
            ]
            selected_idx = st.selectbox(
                "Recent runs",
                options=list(range(len(txns))),
                index=0,
                format_func=lambda i: labels[i]
            )
            selected = txns[selected_idx]
            st.write("Selected ID:", selected.get("id", ""))
            if st.button("Load this run"):
                st.session_state["loaded_guidelines"] = selected.get("guidelines", "")
                st.session_state["loaded_input_code"] = selected.get("input_code", "")
                st.session_state["loaded_output_code"] = selected.get("output_code", "")
                st.session_state["loaded_model"] = selected.get("model", "")
                st.session_state["loaded_runtime_s"] = selected.get("runtime_s", 0)
                st.session_state["loaded_summary"] = ""  
                st.session_state["just_generated"] = False
                #st.experimental_rerun()
    else:
        st.caption("Supabase disabled.")

# -------------------------
# Inputs with upload support
# -------------------------
# Guidelines
guidelines_col, code_col = st.columns(2)

with guidelines_col:
    st.subheader("Guidelines")
    uploaded_guidelines = st.file_uploader("Upload guidelines (.txt, .md)", type=["txt", "md"], key="guidelines_upl", accept_multiple_files=False)
    default_guidelines = st.session_state.get("loaded_guidelines", "")
    if uploaded_guidelines is not None:
        try:
            default_guidelines = uploaded_guidelines.read().decode("utf-8", errors="ignore")
        except Exception:
            st.error("Failed to read guidelines file; fallback to text area.")
    guidelines_text = st.text_area("Or paste guidelines", value=default_guidelines, height=200, placeholder="e.g., optimize loops, add comments, use const where appropriate")

with code_col:
    st.subheader("C code")
    uploaded_code = st.file_uploader("Upload C code (.c, .h, .txt)", type=["c", "h", "txt"], key="code_upl", accept_multiple_files=False)
    default_input_code = st.session_state.get("loaded_input_code", "")
    if uploaded_code is not None:
        try:
            default_input_code = uploaded_code.read().decode("utf-8", errors="ignore")
        except Exception:
            st.error("Failed to read code file; fallback to text area.")
    input_code = st.text_area("Or paste C code", value=default_input_code, height=200, placeholder="Paste C code here")

# -------------------------
# Prompt and generators
# -------------------------
def build_prompt_for_summary_and_code(guidelines: str, code: str, Laungauge:str) -> str:
    return f"""You are an expert {Laungauge} engineer. Apply the following guidelines to rewrite the given {Laungauge} code.
Return a compact JSON object with exactly two keys:
- "summary": a bullet-style explanation of the changes applied (no code blocks), focusing on what changed and why and keep it short to the point.
- "code": the final rewritten {Laungauge} source as a plain string.

Guidelines:
{guidelines}

Original Code:
{code}

Output format example (do not add extra keys):
{{
  "summary": "- Change 1...\\n- Change 2...",
  "code": "#include <...>\\nint main(){{...}}"
}}
"""

def call_ollama(prompt: str, model: str, temperature: float, max_tokens: int, request_timeout: int = 600) -> tuple[str, int]:
    t0 = time.time()
    resp = requests.post(
        f"{os.getenv('OLLAMA_URL', DEFAULT_OLLAMA_URL)}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False
        },
        timeout=request_timeout
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", ""), int((time.time() - t0))

def call_codestral(prompt: str, model: str, temperature: float, max_tokens: int, request_timeout: int = 120) -> tuple[str, int]:
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY not set in environment.")
    t0 = time.time()
    url = f"{MISTRAL_API_URL}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=request_timeout)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return text, int((time.time() - t0))

def generate_text(prompt: str, temperature: float, max_tokens: int, provider_choice: str, model_name: str) -> tuple[str, int]:
    if provider_choice == PROVIDER_OLLAMA:
        return call_ollama(prompt, model_name, temperature, max_tokens)
    else:
        provider_choice == PROVIDER_CODESTRAL
        return call_codestral(prompt, model_name, temperature, max_tokens)

# -------------------------
# Actions
# -------------------------
out_container = st.container()
gen_col1, gen_col2 = st.columns([1, 1])
with gen_col1:
    run_clicked = st.button("Generate changes and code", use_container_width=True)
with gen_col2:
    clear_clicked = st.button("Clear loaded output", use_container_width=True)

if clear_clicked:
    st.session_state["loaded_output_code"] = ""
    st.session_state["loaded_summary"] = ""
    st.session_state["just_generated"] = False
    #st.experimental_rerun()

# Determine selected model string
if provider == PROVIDER_OLLAMA:
    selected_model = ollama_model
else:
    selected_model = codestral_model

def parse_summary_and_code(raw: str) -> tuple[str, str]:
    """
    Attempts to parse a JSON object with 'summary' and 'code'.
    If not valid JSON, fall back to heuristic split:
    - Try to find a JSON block between first '{' and last '}'.
    """
    # First try: direct JSON parse
    try:
        obj = json.loads(raw)
        summary = obj.get("summary", "").strip()
        code = obj.get("code", "").strip()
        return summary, code
    except Exception:
        pass
    # Heuristic: find JSON-like segment
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(raw[start:end+1])
            summary = obj.get("summary", "").strip()
            code = obj.get("code", "").strip()
            return summary, code
        except Exception:
            pass
    # Fallback: put everything into summary, code empty
    return raw.strip(), ""

if run_clicked:
    if not input_code.strip():
        st.error("Please provide C code (paste or upload).")
    elif provider == PROVIDER_CODESTRAL and not MISTRAL_API_KEY:
        st.error("MISTRAL_API_KEY is not set. Configure it to use Codestral.")
    else:
        full_prompt = build_prompt_for_summary_and_code(guidelines_text, input_code, Laungauge)
        with st.spinner(f"Generating with {provider}..."):
            try:
                raw_text, runtime_s = generate_text(
                    full_prompt, temperature, max_tokens, provider_choice=provider, model_name=selected_model
                )
                summary, rewritten_code = parse_summary_and_code(raw_text)
                st.session_state["just_generated"] = True
                st.session_state["loaded_summary"] = summary
                st.session_state["loaded_output_code"] = rewritten_code

                with out_container:
                    st.subheader("What changed")
                    if summary:
                        st.markdown(summary)
                    else:
                        st.info("No summary returned.")

                    # Download button for the code
                    if rewritten_code:
                        st.download_button(
                            label="Download rewritten code",
                            data=rewritten_code.encode("utf-8"),
                            file_name="rewritten.c",
                            mime="text/x-csrc",
                            use_container_width=True
                        )
                    else:
                        st.warning("Model did not return code. Try increasing max tokens or adjusting guidelines.")

                # Save to Supabase (stores code; summary not stored by default)
                rec_id = save_transaction(
                    guidelines=guidelines_text,
                    input_code=input_code,
                    output_code=rewritten_code,
                    model=f"{provider}:{selected_model}",
                    runtime_s=runtime_s,
                    summary = summary
                )
                if rec_id:
                    st.success(f"Saved to history (id: {rec_id})")
                else:
                    if supabase:
                        st.warning("Could not save to Supabase.")
                    else:
                        st.info("Supabase not configured; skipping save.")
            except requests.HTTPError as he:
                st.error(f"HTTP error: {he}")
            except requests.ConnectionError:
                if provider == PROVIDER_OLLAMA:
                    st.error("Cannot connect to Ollama. Ensure it is running and the model is available.")
                else:
                    st.error("Cannot reach Codestral API. Check connectivity or API URL.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# -------------------------
# Loaded output display (from history)
# -------------------------
if st.session_state.get("loaded_output_code") and not st.session_state.get("just_generated"):
    with st.expander("Loaded summary (if any) and code download", expanded=True):
        # Summary isn't stored; show a note
        #st.caption("History currently stores code only; summary won鈥檛 appear unless added to schema.")
        st.download_button(
            label="Download loaded code",
            data=st.session_state["loaded_output_code"].encode("utf-8"),
            file_name="rewritten.c",
            mime="text/x-csrc",
            use_container_width=True
        )
    if st.session_state.get("loaded_model"):
        st.caption(f"Loaded model: {st.session_state['loaded_model']} {st.session_state.get('loaded_runtime_s', 0)} ms")

# -------------------------
# Footer
# -------------------------
if provider == PROVIDER_OLLAMA:
    st.markdown(
        f'<div class="small-caption">Provider: Ollama Model: {selected_model} URL: {os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)}</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div class="small-caption">Provider: Codestral Model: {selected_model} API URL: {MISTRAL_API_URL}</div>',
        unsafe_allow_html=True
    )

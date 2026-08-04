"""
app.py — Model-comparison research pipeline (deployable to Streamlit Cloud).

This is the culmination of everything built so far:
  - Two-agent pipeline (Researcher -> Critic -> Reviser)
  - Running on OpenRouter (open-weight OR closed models, one API)
  - With TWO tools: web_search + fetch_page (read full pages)
  - NEW: run the SAME question through SEVERAL models side by side, so you
    can directly compare answer quality, citations, and speed.

The design change that makes comparison possible: `model` is now a PARAMETER
threaded through every function, instead of a global constant. That single
change is what lets one run drive many models.

SECRETS NEEDED (set locally as env vars, or in Streamlit Cloud -> Secrets):
    OPENROUTER_API_KEY   — get free at openrouter.ai/keys
    TAVILY_API_KEY       — get free at tavily.com
"""

import os
import json
import time
import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI
from tavily import TavilyClient


# ---------------------------------------------------------------------------
# SECRETS — works locally (env vars) and on Streamlit Cloud (secrets store).
# ---------------------------------------------------------------------------
for key_name in ("OPENROUTER_API_KEY", "TAVILY_API_KEY"):
    if key_name not in os.environ:
        try:
            os.environ[key_name] = st.secrets[key_name]
        except Exception:
            pass

missing = [k for k in ("OPENROUTER_API_KEY", "TAVILY_API_KEY") if not os.environ.get(k)]
if missing:
    st.error(
        f"Missing API key(s): {', '.join(missing)}. Set them as environment "
        "variables locally, or in the app's Secrets settings on Streamlit Cloud."
    )
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
tavily_client = TavilyClient()


# ===========================================================================
# TOOLS  (web_search via Tavily, fetch_page via requests+BeautifulSoup)
# ===========================================================================
def web_search_tool(query: str) -> str:
    try:
        response = tavily_client.search(query=query, max_results=4, search_depth="basic")
    except Exception as e:
        return f"Search failed: {e}"
    results = response.get("results", [])
    if not results:
        return f"No results found for '{query}'."
    return "\n\n".join(
        f"- {r.get('title', 'Untitled')}\n  {r.get('content', '')}\n  URL: {r.get('url', '')}"
        for r in results
    )


def fetch_page_tool(url: str, max_chars: int = 6000) -> str:
    try:
        resp = requests.get(url, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0 (research-agent)"})
        resp.raise_for_status()
    except Exception as e:
        return f"Could not fetch {url}: {e}"
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "title"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines()]
    clean = "\n".join(ln for ln in lines if ln)
    if len(clean) > max_chars:
        clean = clean[:max_chars] + "\n\n[...truncated...]"
    return f"Full text of {url}:\n\n{clean}"


tools = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web. Returns titles, snippets, and URLs. Use first.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "The search query"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_page",
        "description": "Fetch the full readable text of a web page by URL. Use after "
                       "web_search to read a promising source in depth.",
        "parameters": {"type": "object",
                       "properties": {"url": {"type": "string", "description": "The full URL to fetch"}},
                       "required": ["url"]}}},
]


def execute_tool(name, args):
    if name == "web_search":
        return web_search_tool(args.get("query", ""))
    if name == "fetch_page":
        return fetch_page_tool(args.get("url", ""))
    return f"Unknown tool: {name}"


# ===========================================================================
# AGENT LOOP — note `model` is now the FIRST parameter. This is the change
# that makes model comparison possible.
# ===========================================================================
def run_agent_loop(model, system_prompt, user_content, use_tools=True, max_steps=8):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    for step in range(max_steps):
        is_last = (step == max_steps - 1)
        kwargs = {"model": model, "messages": messages, "max_tokens": 2000}
        if use_tools and not is_last:
            kwargs["tools"] = tools

        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        if not msg.tool_calls:
            text = msg.content or ""
            if text.strip():
                return text
            messages.append({"role": "user",
                             "content": "You returned nothing. Write your answer now as plain text."})
            retry = client.chat.completions.create(model=model, messages=messages, max_tokens=2000)
            return retry.choices[0].message.content or "(no answer produced)"

        messages.append(msg)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "(agent could not produce a final answer)"


RESEARCHER_PROMPT = (
    "You are a research assistant. First use web_search to find sources, then "
    "use fetch_page to read the most promising one or two in full before writing. "
    "Produce a clear, well-supported answer with inline source URLs."
)
CRITIC_PROMPT = (
    "You are a sharp, skeptical fact-checker. You are given a QUESTION and a DRAFT "
    "ANSWER by someone else. Your ONLY job is to critique the draft -- do NOT rewrite "
    "it or answer yourself. List specific problems: unsupported claims, factual "
    "errors, vague statements, missing context. Numbered list, concise."
)
REVISER_PROMPT = (
    "You are revising your own earlier draft. You are given the QUESTION, your DRAFT, "
    "and a CRITIQUE. Use web_search and fetch_page to fill the gaps, then produce an "
    "improved FINAL answer with source URLs. Output only the final answer."
)


def run_for_model(model, question, full_pipeline=True):
    """Run one model and return its result + timing, catching errors so one
    failing model doesn't break the whole comparison."""
    start = time.time()
    try:
        draft = run_agent_loop(model, RESEARCHER_PROMPT, question)
        if not full_pipeline:
            return {"final": draft, "draft": None, "critique": None,
                    "elapsed": time.time() - start, "error": None}

        critic_input = f"QUESTION:\n{question}\n\nDRAFT ANSWER:\n{draft}"
        critique = run_agent_loop(model, CRITIC_PROMPT, critic_input, use_tools=False)

        reviser_input = f"QUESTION:\n{question}\n\nYOUR DRAFT:\n{draft}\n\nCRITIQUE:\n{critique}"
        final = run_agent_loop(model, REVISER_PROMPT, reviser_input)

        return {"final": final, "draft": draft, "critique": critique,
                "elapsed": time.time() - start, "error": None}
    except Exception as e:
        return {"final": None, "draft": None, "critique": None,
                "elapsed": time.time() - start, "error": str(e)}


# ===========================================================================
# UI
# ===========================================================================
st.set_page_config(page_title="Model Comparison — Research Pipeline",
                   page_icon="⚖️", layout="wide")
st.title("⚖️ Research Pipeline — Model Comparison")
st.caption("Run the same question through several models side by side. "
           "Each runs the Researcher → Critic → Reviser pipeline with web search + page fetch.")

question = st.text_area(
    "Research question:",
    placeholder="e.g. Do solar farms harm wildlife, and how can that be mitigated?",
    height=90,
)

st.markdown("**Models to compare** (one slug per line). Browse slugs at "
            "[openrouter.ai/models](https://openrouter.ai/models). "
            "`openrouter/free` auto-picks a working free model; paid slugs need credits.")
models_text = st.text_area(
    "Models:",
    value="openrouter/free\nopenai/gpt-oss-120b",
    height=90,
    label_visibility="collapsed",
)

col_a, col_b = st.columns([1, 2])
with col_a:
    full_pipeline = st.toggle("Full pipeline", value=False,
                              help="On: Researcher→Critic→Reviser (slower, ~3x the calls). "
                                   "Off: Researcher only (faster, good for quick comparisons).")
with col_b:
    st.caption("Tip: start with 'Researcher only' to compare quickly and save "
               "quota. Each full-pipeline run is a dozen-plus API calls per model.")

if st.button("Compare models", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
        st.stop()

    models = [m.strip() for m in models_text.splitlines() if m.strip()]
    if not models:
        st.warning("Please enter at least one model slug.")
        st.stop()

    st.divider()
    columns = st.columns(len(models))

    for col, model in zip(columns, models):
        with col:
            st.subheader(model)
            with st.spinner(f"Running {model}…"):
                result = run_for_model(model, question, full_pipeline=full_pipeline)

            if result["error"]:
                st.error(f"Failed: {result['error']}")
                continue

            st.caption(f"⏱️ {result['elapsed']:.1f}s")
            st.markdown(result["final"])

            if full_pipeline and result["draft"] is not None:
                with st.expander("First draft"):
                    st.markdown(result["draft"])
                with st.expander("Critique"):
                    st.markdown(result["critique"])

    st.divider()
    st.caption("⚠️ Verify citations — smaller/free models sometimes fabricate "
               "plausible-looking sources. Click each link to confirm it resolves.")

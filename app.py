"""
app.py — Web version of the two-agent pipeline (Researcher -> Critic -> Reviser).

This is the SAME pipeline logic from agent_pipeline.py, with two changes
that make it cloud-deployable:

  1. The terminal input()/print() loop is replaced by a Streamlit web UI
     (a text box, a button, and live status updates). Cloud apps have no
     terminal to type into — they have a web page instead.

  2. API keys are read from Streamlit secrets OR environment variables,
     so the SAME code runs locally (env vars) and on the cloud (secrets),
     without ever hardcoding a key into the file.

Run locally with:   streamlit run app.py
Deploy by pushing this + requirements.txt to GitHub, then deploying on
Streamlit Community Cloud (free).
"""

import os
import streamlit as st
from anthropic import Anthropic
from tavily import TavilyClient

MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# API KEYS — works in BOTH environments.
# Locally: reads the environment variables you already set with setx.
# On Streamlit Cloud: reads the secrets you paste into the deploy dialog.
# st.secrets behaves like a dict; we copy any keys it has into os.environ
# so the Anthropic and Tavily SDKs (which read env vars) pick them up.
# ---------------------------------------------------------------------------
for key_name in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY"):
    if key_name not in os.environ:
        try:
            os.environ[key_name] = st.secrets[key_name]
        except Exception:
            pass  # not in secrets either; we'll catch the missing key below

# Fail clearly if keys are missing, instead of a cryptic SDK error.
missing = [k for k in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY") if not os.environ.get(k)]
if missing:
    st.error(
        f"Missing API key(s): {', '.join(missing)}. "
        "Set them as environment variables locally, or in the app's "
        "Secrets settings on Streamlit Cloud."
    )
    st.stop()

client = Anthropic()
tavily_client = TavilyClient()


# ===========================================================================
# PIPELINE LOGIC — unchanged from agent_pipeline.py, except print() calls
# are replaced with an optional status callback so the web UI can show
# progress. The agent reasoning is identical.
# ===========================================================================
def web_search_tool(query: str) -> str:
    try:
        response = tavily_client.search(query=query, max_results=3, search_depth="basic")
    except Exception as e:
        return f"Search failed: {e}"
    results = response.get("results", [])
    if not results:
        return f"No results found for '{query}'."
    return "\n\n".join(
        f"- {r.get('title', 'Untitled')}\n  {r.get('content', '')}\n  Source: {r.get('url', '')}"
        for r in results
    )


tools = [
    {
        "name": "web_search",
        "description": "Search the web for current information on a topic.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
        },
    }
]


def run_agent_loop(system_prompt, user_content, use_tools=True, max_steps=8,
                   max_tokens=2000, label="AGENT", status=None):
    """`status` is an optional function(str) the UI uses to show progress."""
    def report(msg):
        if status:
            status(msg)

    messages = [{"role": "user", "content": user_content}]

    for step in range(max_steps):
        is_last_step = (step == max_steps - 1)
        allow_tools = use_tools and not is_last_step

        kwargs = {"model": MODEL, "max_tokens": max_tokens,
                  "system": system_prompt, "messages": messages}
        if allow_tools:
            kwargs["tools"] = tools

        response = client.messages.create(**kwargs)
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if not tool_calls:
            text = "".join(b.text for b in response.content if b.type == "text")
            if text.strip():
                return text
            report(f"[{label}] empty response (stop_reason={response.stop_reason}); retrying.")
            messages.append({"role": "user",
                             "content": "You returned an empty response. Please write your answer now as plain text."})
            retry = client.messages.create(model=MODEL, max_tokens=max_tokens,
                                           system=system_prompt, messages=messages)
            retry_text = "".join(b.text for b in retry.content if b.type == "text")
            if retry_text.strip():
                return retry_text
            return f"(no answer produced; stop_reason={retry.stop_reason})"

        tool_results = []
        for call in tool_calls:
            q = call.input.get("query", "")
            report(f"[{label}] searching: {q}")
            result = web_search_tool(call.input["query"]) if call.name == "web_search" else "Unknown tool"
            tool_results.append({"type": "tool_result", "tool_use_id": call.id, "content": result})
        messages.append({"role": "user", "content": tool_results})

    return "(agent could not produce a final answer)"


RESEARCHER_PROMPT = (
    "You are a research assistant. Use the web_search tool to gather facts, "
    "then write a clear, well-supported answer to the user's question. "
    "Cite sources inline where you can."
)
CRITIC_PROMPT = (
    "You are a sharp, skeptical fact-checker. You will be given a QUESTION "
    "and a DRAFT ANSWER written by someone else. Your ONLY job is to critique "
    "the draft -- do NOT rewrite it or answer the question yourself. "
    "List specific problems: unsupported claims, factual errors, vague "
    "statements, missing context, or gaps. Be concise and use a numbered list."
)
REVISER_PROMPT = (
    "You are a research assistant revising your own earlier draft. You will "
    "be given the original QUESTION, your DRAFT, and a CRITIQUE of that draft. "
    "Use the web_search tool to fill any gaps the critique identified, then "
    "produce an improved FINAL answer that addresses the critique. Output only "
    "the final answer."
)


# ===========================================================================
# THE WEB UI — this replaces the old `if __name__ == "__main__"` input loop.
# ===========================================================================
st.set_page_config(page_title="Two-Agent Research Pipeline", page_icon="🔎")
st.title("🔎 Two-Agent Research Pipeline")
st.caption("Researcher → Critic → Reviser. Each stage is a separate AI agent.")


def signed_in_as():
    """
    Return a human-readable 'who is signed in' string, degrading gracefully.

    IMPORTANT: since Streamlit 1.42, st.user only exposes a viewer's email
    if you've configured a full identity provider (Google OIDC) in secrets.
    With the plain viewer allow-list (no OIDC), st.user has no email -- so
    we must NOT assume st.user.email exists, or the app crashes. This helper
    tries several safe paths and falls back to a neutral message.
    """
    try:
        user = st.user  # may be an empty dict-like object
        # .get works because st.user inherits from dict
        email = user.get("email") if hasattr(user, "get") else None
        if email:
            return f"Signed in as **{email}**"
        name = user.get("name") if hasattr(user, "get") else None
        if name:
            return f"Signed in as **{name}**"
    except Exception:
        pass
    # No identifiable info available (allow-list without OIDC, or running
    # locally). Still confirms the page loaded for an authenticated viewer.
    return "Signed in via Streamlit (email shown only with full OIDC auth)"


st.caption("👤 " + signed_in_as())

question = st.text_area(
    "Ask a research question:",
    placeholder="e.g. Is nuclear power a safe and viable way to reduce carbon emissions?",
    height=100,
)

if st.button("Run pipeline", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
        st.stop()

    # st.status shows a live, collapsible progress panel while agents work.
    with st.status("Running the pipeline…", expanded=True) as status_box:
        def show(msg):
            status_box.write(msg)

        show("**Stage 1 — Researcher** drafting an answer…")
        draft = run_agent_loop(RESEARCHER_PROMPT, question, use_tools=True,
                               label="RESEARCHER", status=show)

        show("**Stage 2 — Critic** reviewing the draft…")
        critic_input = f"QUESTION:\n{question}\n\nDRAFT ANSWER:\n{draft}"
        critique = run_agent_loop(CRITIC_PROMPT, critic_input, use_tools=False,
                                  label="CRITIC", status=show)

        show("**Stage 3 — Reviser** producing the final answer…")
        reviser_input = f"QUESTION:\n{question}\n\nYOUR DRAFT:\n{draft}\n\nCRITIQUE OF YOUR DRAFT:\n{critique}"
        final = run_agent_loop(REVISER_PROMPT, reviser_input, use_tools=True,
                               max_tokens=4000, label="REVISER", status=show)

        status_box.update(label="Done!", state="complete", expanded=False)

    # Show the final answer prominently, with the intermediate stages tucked
    # into expanders so you can inspect how the agents improved the answer.
    st.subheader("Final answer")
    st.markdown(final)

    with st.expander("See the researcher's first draft"):
        st.markdown(draft)
    with st.expander("See the critic's critique"):
        st.markdown(critique)

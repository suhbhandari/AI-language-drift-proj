"""
Live demo: paste any text, get an AI-probability read.

Run locally:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your_key_here
    streamlit run app/streamlit_app.py

Or deploy free on Streamlit Community Cloud / Hugging Face Spaces
(add ANTHROPIC_API_KEY as a secret in either platform's settings).
"""

import os
import re
import streamlit as st
import anthropic

st.set_page_config(page_title="AI Language Drift — Live Demo", page_icon="🔍", layout="centered")

st.title("Is this text AI-written?")
st.caption(
    "Live demo from the r/AskReddit AI-language-drift project. "
    "Paste any text below and get an AI-probability read from an LLM judge, "
    "plus the same regex-based linguistic markers used in the original analysis."
)

# ── Linguistic patterns from the original notebook ──────────────────────────
PATTERNS = {
    'Additive markers (additionally, furthermore...)': re.compile(
        r'\b(additionally|furthermore|moreover|firstly|secondly|'
        r'lastly|in addition|on the other hand)\b', re.IGNORECASE),
    'Hedging (it is important to, one might...)': re.compile(
        r'\b(it is important to|it should be noted|it is worth|'
        r'one might|one could argue|it can be|this could be|'
        r'it is essential)\b', re.IGNORECASE),
    'Passive voice': re.compile(
        r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', re.IGNORECASE),
    'Not X but Y structure': re.compile(
        r'\bnot\b.{1,40}\bbut\b', re.IGNORECASE),
    'Conclusion phrases (in conclusion, overall...)': re.compile(
        r'\b(in conclusion|in summary|to summarize|overall|'
        r'ultimately|at the end of the day|all in all)\b', re.IGNORECASE),
    'Affirmations (certainly, absolutely...)': re.compile(
        r'\b(certainly|absolutely|of course|definitely|'
        r'undoubtedly|without a doubt)\b', re.IGNORECASE),
}


def get_llm_judgment(text: str, api_key: str):
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""Rate how likely the following text was written by an AI language model
versus a human, on a scale of 0-100 (0 = definitely human, 100 = definitely AI).
Then give a one-sentence reason.

Respond in exactly this format:
SCORE: <number>
REASON: <one sentence>

Text: "{text[:1500]}"
"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    reply = response.content[0].text.strip()
    score_match = re.search(r'SCORE:\s*(\d+)', reply)
    reason_match = re.search(r'REASON:\s*(.+)', reply)
    score = int(score_match.group(1)) if score_match else None
    reason = reason_match.group(1).strip() if reason_match else reply
    return score, reason


# ── Input ─────────────────────────────────────────────────────────────────
text_input = st.text_area(
    "Paste text to analyze",
    height=180,
    placeholder="Paste a paragraph, comment, or essay excerpt here...",
)

api_key = os.environ.get("ANTHROPIC_API_KEY") or st.sidebar.text_input(
    "Anthropic API key", type="password",
    help="Set ANTHROPIC_API_KEY as an environment variable, or paste one here for this session only."
)

analyze = st.button("Analyze", type="primary", disabled=not text_input.strip())

if analyze:
    if not api_key:
        st.error("Add an Anthropic API key in the sidebar to run the LLM judge.")
    else:
        with st.spinner("Scoring..."):
            try:
                score, reason = get_llm_judgment(text_input, api_key)
            except Exception as e:
                st.error(f"API call failed: {e}")
                score, reason = None, None

        if score is not None:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("AI-likelihood score", f"{score}/100")
            with col2:
                st.write(f"**Reasoning:** {reason}")

        st.divider()
        st.subheader("Linguistic markers detected")
        st.caption("Same regex patterns from the original r/AskReddit analysis.")

        any_hits = False
        for name, pattern in PATTERNS.items():
            hits = pattern.findall(text_input)
            if hits:
                any_hits = True
                st.write(f"✅ **{name}** — {len(hits)} match(es)")
        if not any_hits:
            st.write("No AI-typical patterns detected in this text.")

st.divider()
st.caption(
    "Part of a larger project analyzing whether r/AskReddit language has shifted "
    "toward AI-typical patterns since ChatGPT's November 2022 launch. "
    "[Full write-up and methodology on GitHub]."
)

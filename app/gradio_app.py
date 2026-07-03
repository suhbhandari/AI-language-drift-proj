"""
Live demo: paste any text, get an AI-probability read.

Run locally:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your_key_here
    python app/gradio_app.py

Or deploy free on Hugging Face Spaces:
    1. Create a new Space (SDK: Gradio)
    2. Upload this file as app.py, plus requirements.txt
    3. Add ANTHROPIC_API_KEY under Space Settings -> Secrets
"""

import os
import re
import gradio as gr
import anthropic

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

ENV_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


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


def analyze(text: str, api_key_input: str):
    key = ENV_API_KEY or api_key_input
    if not text or not text.strip():
        return "Paste some text first.", "", ""
    if not key:
        return "Add an Anthropic API key below to run the LLM judge.", "", ""

    try:
        score, reason = get_llm_judgment(text, key)
    except Exception as e:
        return f"API call failed: {e}", "", ""

    score_md = f"## {score}/100 AI-likelihood" if score is not None else "Score unavailable"
    reason_md = f"**Reasoning:** {reason}"

    pattern_lines = []
    for name, pattern in PATTERNS.items():
        hits = pattern.findall(text)
        if hits:
            pattern_lines.append(f"- ✅ **{name}** — {len(hits)} match(es)")
    pattern_md = "\n".join(pattern_lines) if pattern_lines else "No AI-typical patterns detected in this text."

    return score_md, reason_md, pattern_md


with gr.Blocks(title="AI Language Drift — Live Demo") as demo:
    gr.Markdown("# Is this text AI-written?")
    gr.Markdown(
        "Live demo from the r/AskReddit AI-language-drift project. "
        "Paste any text below and get an AI-probability read from an LLM judge, "
        "plus the same regex-based linguistic markers used in the original analysis."
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Paste text to analyze",
                lines=8,
                placeholder="Paste a paragraph, comment, or essay excerpt here...",
            )
            api_key_box = gr.Textbox(
                label="Anthropic API key (only needed if not set as an environment variable)",
                type="password",
                visible=ENV_API_KEY is None,
            )
            analyze_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=2):
            score_output = gr.Markdown()
            reason_output = gr.Markdown()
            gr.Markdown("---")
            gr.Markdown("### Linguistic markers detected")
            gr.Markdown("Same regex patterns from the original r/AskReddit analysis.")
            patterns_output = gr.Markdown()

    analyze_btn.click(
        fn=analyze,
        inputs=[text_input, api_key_box],
        outputs=[score_output, reason_output, patterns_output],
        api_name="predict",
    )

    gr.Markdown(
        "---\n"
        "Part of a larger project analyzing whether r/AskReddit language has shifted "
        "toward AI-typical patterns since ChatGPT's November 2022 launch. "
        "[Full write-up and methodology on GitHub]."
    )

if __name__ == "__main__":
    demo.launch() 

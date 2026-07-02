# ══════════════════════════════════════════════════════════════════════════
# TIER 1 UPGRADE — Embedding-based classifier + LLM-as-judge
# Paste these as new cells AFTER your existing pipeline (needs df_labeled_sample,
# df_reddit, and CUTOFF already defined from the earlier cells)
#
# NOTE ON KAGGLE INTERNET: sentence-transformers needs to download model weights
# the first time. In Kaggle: Settings (right sidebar) → Internet → On.
# If Internet must stay off (e.g. for a competition), instead attach a Kaggle
# "Model" (search "all-MiniLM-L6-v2" in Kaggle Models) and load from that path.
# ══════════════════════════════════════════════════════════════════════════

# ── CELL A: Install + load embedding model ──────────────────────────────────
!pip install -q sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading embedding model (all-MiniLM-L6-v2, 384-dim, fast + strong baseline)...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')


# ── CELL B: Re-train classifier on embeddings instead of TF-IDF ─────────────
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

print("Embedding labeled training text (this takes a few minutes on 20k rows)...")
X_text = df_labeled_sample['text'].astype(str).tolist()
X_embeddings = embed_model.encode(X_text, batch_size=64, show_progress_bar=True)

X_train_emb, X_test_emb, y_train_emb, y_test_emb = train_test_split(
    X_embeddings, df_labeled_sample['generated'],
    test_size=0.2, random_state=42
)

clf_embed = LogisticRegression(max_iter=1000)
clf_embed.fit(X_train_emb, y_train_emb)
y_pred_emb = clf_embed.predict(X_test_emb)

print("\nEmbedding-based classifier performance:")
print(classification_report(y_test_emb, y_pred_emb, target_names=['Human', 'AI']))
embed_accuracy = accuracy_score(y_test_emb, y_pred_emb)
print(f"Accuracy: {embed_accuracy:.4f}")
print(f"(Compare to TF-IDF classifier accuracy: {accuracy_score(y_test, y_pred):.4f})")


# ── CELL C: Score Reddit comments with the embedding classifier ─────────────
print("Embedding Reddit comments...")
reddit_embeddings = embed_model.encode(
    df_reddit['body'].astype(str).tolist(),
    batch_size=64, show_progress_bar=True
)
df_reddit['ai_probability_embed'] = clf_embed.predict_proba(reddit_embeddings)[:, 1]

pre_embed  = df_reddit[df_reddit['date'] <  CUTOFF]['ai_probability_embed']
post_embed = df_reddit[df_reddit['date'] >= CUTOFF]['ai_probability_embed']

from scipy import stats
t_stat_embed, p_value_embed = stats.ttest_ind(pre_embed, post_embed)
mean_pre_embed  = pre_embed.mean()
mean_post_embed = post_embed.mean()
pct_change_embed = ((mean_post_embed - mean_pre_embed) / mean_pre_embed) * 100

print(f"\n[Embedding classifier] Pre-ChatGPT mean : {mean_pre_embed:.4f}")
print(f"[Embedding classifier] Post-ChatGPT mean: {mean_post_embed:.4f}")
print(f"[Embedding classifier] Change           : {pct_change_embed:+.2f}%")
print(f"[Embedding classifier] p-value           : {p_value_embed:.6f}")
print(f"\n[TF-IDF classifier]    Change from earlier cells: {pct_change:+.2f}% (p={p_value:.4f})")
print("\nIf these two independent methods agree in direction, that's a much")
print("stronger claim than either one alone. If they disagree, that's worth")
print("discussing too, it means the earlier finding may be method-dependent.")


# ── CELL D: LLM-as-judge on a sample (independent second signal) ────────────
# This uses the Anthropic API. Add your API key as a Kaggle Secret
# (Add-ons → Secrets → ANTHROPIC_API_KEY), don't hardcode it.

!pip install -q anthropic

import anthropic
from kaggle_secrets import UserSecretsClient
import time
import json

user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

def llm_judge_ai_probability(text, retries=3):
    """Ask an LLM to rate how likely a text is AI-generated, 0-100.
    This is a genuinely independent signal from the trained classifiers above,
    since it's not trained on the same labeled dataset."""
    prompt = f"""Rate how likely the following text was written by an AI language model
versus a human, on a scale of 0-100 (0 = definitely human, 100 = definitely AI).
Consider things like: overly polished phrasing, hedge words, generic structure,
lack of typos/slang/casual tone for the context. Respond with ONLY a number, nothing else.

Text: "{text[:500]}"
"""
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            score_text = response.content[0].text.strip()
            return float(''.join(c for c in score_text if c.isdigit() or c == '.'))
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed after {retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)

# IMPORTANT: sample a small subset. This costs API calls per comment.
# 300-500 comments (balanced pre/post) is enough for a t-test and keeps cost low.
SAMPLE_SIZE_PER_GROUP = 200

pre_sample  = df_reddit[df_reddit['date'] <  CUTOFF].sample(
    n=min(SAMPLE_SIZE_PER_GROUP, len(df_reddit[df_reddit['date'] < CUTOFF])),
    random_state=42
)
post_sample = df_reddit[df_reddit['date'] >= CUTOFF].sample(
    n=min(SAMPLE_SIZE_PER_GROUP, len(df_reddit[df_reddit['date'] >= CUTOFF])),
    random_state=42
)

print(f"Scoring {len(pre_sample)} pre-ChatGPT comments with LLM judge...")
pre_sample = pre_sample.copy()
pre_sample['llm_ai_score'] = pre_sample['body'].apply(llm_judge_ai_probability)

print(f"Scoring {len(post_sample)} post-ChatGPT comments with LLM judge...")
post_sample = post_sample.copy()
post_sample['llm_ai_score'] = post_sample['body'].apply(llm_judge_ai_probability)

# drop failed calls
pre_scores  = pre_sample['llm_ai_score'].dropna()
post_scores = post_sample['llm_ai_score'].dropna()

t_stat_llm, p_value_llm = stats.ttest_ind(pre_scores, post_scores)
print(f"\n[LLM-as-judge] Pre-ChatGPT mean score  : {pre_scores.mean():.2f}/100")
print(f"[LLM-as-judge] Post-ChatGPT mean score : {post_scores.mean():.2f}/100")
print(f"[LLM-as-judge] p-value                  : {p_value_llm:.6f}")


# ── CELL E: Three-way comparison plot ────────────────────────────────────────
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
methods = ['TF-IDF\nClassifier', 'Embedding\nClassifier', 'LLM-as-Judge\n(sampled)']
pre_vals  = [mean_pre, mean_pre_embed, pre_scores.mean() / 100]
post_vals = [mean_post, mean_post_embed, post_scores.mean() / 100]

x = np.arange(len(methods))
width = 0.35
ax.bar(x - width/2, pre_vals,  width, label='Pre-ChatGPT', color='steelblue', edgecolor='black')
ax.bar(x + width/2, post_vals, width, label='Post-ChatGPT', color='tomato', edgecolor='black')
ax.set_title('AI-Probability: Three Independent Methods, Pre vs Post ChatGPT',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Mean AI probability (normalized 0-1)')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
plt.tight_layout()
plt.savefig('/kaggle/working/three_method_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: three_method_comparison.png")

print("""
WHY THIS MATTERS FOR THE WRITEUP:
A single classifier finding a small effect could be noise or method artifact.
Three independent methods (bag-of-words, embeddings, and an LLM with no
training on this specific task) pointing the same direction is a much
stronger claim. If they disagree, that itself is a finding worth discussing,
it suggests the original result may be sensitive to how "AI-like" is defined.
""")

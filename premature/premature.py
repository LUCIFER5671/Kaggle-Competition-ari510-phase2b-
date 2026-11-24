"""ARI 410/510 Phase 2b - Detecting Premature Problem Solving (Roshkrishna Ranjith, Team 9)

Label 1 = premature problem solving (the response moves into advice/solutions before adequately
acknowledging the emotional concern); 0 = it acknowledges the concern first.

Usage (each command prints dev scores and writes submissions/<run>.csv):
    python premature.py run1   # TF-IDF + logistic regression
    python premature.py run2   # MiniLM embeddings (context, response opening, full response) + MLP (PyTorch)
    python premature.py run3   # Claude zero-shot, short prompt
    python premature.py run4   # Claude zero-shot, host's annotation rules + examples + "focus on the opening"
    python premature.py run5   # Claude few-shot: run4 prompt + 12 labeled train examples

Dev evaluation:
  * "val" - fit on train.csv (120 rows), score on validation.csv (30 rows). Used to compare ALL runs.
  * "cv5" - stratified 5-fold cross-validation over all 150 labeled rows (trainable runs only; more stable).
Final test predictions from trainable runs use a model fit on all 150 labeled rows.
Claude runs read ANTHROPIC_API_KEY (falls back to the value saved with `setx` on Windows).
"""
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "submissions"
CACHE = HERE / "cache"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def clean(t):
    return re.sub(r"\s+", " ", str(t).replace("\xa0", " ").replace("�", "'")).strip()


def load():
    dfs = [pd.read_csv(DATA / f"{n}.csv") for n in ("train", "validation", "test")]
    for d in dfs:
        d["context"] = d["context"].map(clean)
        d["response"] = d["response"].map(clean)
    return dfs


def opening(text, n_sentences=2):
    """First sentences of the response - where acknowledgement vs advice is decided."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:n_sentences])


def scores(y_true, y_pred):
    return {
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "n": len(y_true),
    }


def write_submission(run, test, preds):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{run}.csv"
    pd.DataFrame({"id": test["id"], "label": [int(p) for p in preds]}).to_csv(path, index=False)
    print(f"wrote {path}")


def report(run, results):
    print(f"\n== {run} dev results ==")
    for k, v in results.items():
        print(f"  {k}: {v}")
    log = HERE / "results.json"
    allres = json.loads(log.read_text()) if log.exists() else {}
    allres[run] = results
    log.write_text(json.dumps(allres, indent=2))


# ---------------------------------------------------------------------------
# Trainable models
# ---------------------------------------------------------------------------

def make_tfidf(C=1.0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    def fit(df):
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
            LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED),
        )
        model.fit(df["response"], df["label"])
        return lambda d: model.predict(d["response"])
    return fit


_EMB = {}


def embed(texts):
    from sentence_transformers import SentenceTransformer
    if "model" not in _EMB:
        _EMB["model"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    missing = list(dict.fromkeys(t for t in texts if t not in _EMB))
    if missing:
        vecs = _EMB["model"].encode(missing, normalize_embeddings=True, show_progress_bar=False)
        for t, v in zip(missing, vecs):
            _EMB[t] = v
    return np.stack([_EMB[t] for t in texts])


def features(df):
    c = embed(list(df["context"]))
    o = embed([opening(r) for r in df["response"]])
    r = embed(list(df["response"]))
    return np.hstack([c, o, r, c * o]).astype(np.float32)


def make_mlp(hidden=64, epochs=150, lr=1e-3, weight_decay=1e-2, dropout=0.3):
    import torch
    from torch import nn

    def fit(df):
        set_seed()
        x = torch.tensor(features(df))
        y = torch.tensor(df["label"].values, dtype=torch.long)
        counts = torch.bincount(y, minlength=2).float().clamp(min=1)
        net = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts))
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss_fn(net(x), y).backward()
            opt.step()
        net.eval()

        def predict(d):
            with torch.no_grad():
                return net(torch.tensor(features(d))).argmax(1).numpy()
        return predict
    return fit


def cv5(fit, labeled):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.zeros(len(labeled), dtype=int)
    for tr, te in skf.split(labeled, labeled["label"]):
        pred[te] = fit(labeled.iloc[tr])(labeled.iloc[te])
    return scores(labeled["label"], pred)


def evaluate_trainable(run, fit, train, val, test, settings):
    set_seed()
    labeled = pd.concat([train, val], ignore_index=True)
    results = {"val": scores(val["label"], fit(train)(val)), "cv5": cv5(fit, labeled), "settings": settings}
    report(run, results)
    write_submission(run, test, fit(labeled)(test))


# ---------------------------------------------------------------------------
# Claude runs
# ---------------------------------------------------------------------------

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = "medium"

SHORT_PROMPT = """You label replies written by counselors to people describing a mental-health concern.

Decide whether the reply shows PREMATURE PROBLEM SOLVING:
- "premature": the reply moves into advice, action steps, recommendations, diagnosis, treatment or other \
directed guidance BEFORE adequately acknowledging the emotional concern the person expressed.
- "acknowledges_first": the reply adequately acknowledges / validates the emotional concern before \
moving to any advice (or gives no advice at all)."""

GUIDELINES_PROMPT = SHORT_PROMPT + """

Annotation rules (from the dataset's guidelines):
| Rule | Label |
| The response adequately acknowledges the emotional concern before moving into advice/action/recommendation/\
diagnosis/treatment/other solution or directed guidance | acknowledges_first |
| The response moves into advice/action/recommendation/diagnosis/treatment/other solution or directed guidance \
before adequately acknowledging the emotional concern | premature |

How to decide:
1. Identify the emotional concern the person explicitly expressed in the context.
2. Read the reply from the start. Find the first point where it gives advice or directed guidance.
3. Ask whether the text BEFORE that point acknowledges the concern: naming or reflecting the feeling, \
normalizing it ("this is more common than we realize, and understandable"), or showing empathy. Acknowledgement \
can be brief; it does not need to be long or elaborate to count.
4. Order matters: empathy that only appears after the advice does not count.

Examples from the guidelines:
- acknowledges_first: "This is actually more common than we often realize, and actually understandable, as he has \
gone through a trauma, an unresolved existential crisis..." (acknowledges with empathy first)
- premature: "I don't know how long ago your breakup was, but I would suggest that you reach out to friends or family \
that you trust and perhaps a local mental health clinician..." (suggests an action without acknowledging the concern first)"""

SCHEMA = {
    "type": "object",
    "properties": {
        "concern": {"type": "string", "description": "The emotional concern the person expressed."},
        "first_advice": {"type": "string", "description": "Where the reply first gives advice or guidance (quote a few words), or 'none'."},
        "acknowledged_before_advice": {"type": "string", "description": "What, if anything, acknowledges the concern before that point."},
        "label": {"type": "string", "enum": ["premature", "acknowledges_first"]},
    },
    "required": ["concern", "first_advice", "acknowledged_before_advice", "label"],
    "additionalProperties": False,
}
LABEL_TO_INT = {"acknowledges_first": 0, "premature": 1}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def ensure_api_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                os.environ["ANTHROPIC_API_KEY"] = winreg.QueryValueEx(k, "ANTHROPIC_API_KEY")[0]
                return
        except OSError:
            pass
    sys.exit("ANTHROPIC_API_KEY is not set.")


def item_block(row):
    return f"<context>\n{row['context']}\n</context>\n\n<reply>\n{row['response']}\n</reply>"


def shots_block(shots):
    parts = [f"<example>\n{item_block(r)}\n<label>{INT_TO_LABEL[int(r['label'])]}</label>\n</example>"
             for _, r in shots.iterrows()]
    return "Labeled examples from the same dataset:\n\n" + "\n\n".join(parts) + "\n\nNow label this item.\n\n"


def claude_label(system, row, shots=None):
    """One Claude call -> (label_int, parsed_json). Cached on disk, so reruns are free and identical."""
    import anthropic
    user = (shots_block(shots) if shots is not None else "") + item_block(row)
    key = hashlib.sha256(f"{MODEL}|{EFFORT}|{system}|{user}".encode()).hexdigest()[:24]
    CACHE.mkdir(exist_ok=True)
    cpath = CACHE / f"{key}.json"
    if cpath.exists():
        out = json.loads(cpath.read_text())
    else:
        resp = anthropic.Anthropic().messages.create(
            model=MODEL,
            max_tokens=4000,
            system=system,
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"refusal on {row['id']}")
        out = json.loads(next(b.text for b in resp.content if b.type == "text"))
        out["_usage"] = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
        cpath.write_text(json.dumps(out, indent=2))
    return LABEL_TO_INT[out["label"]], out


def fewshot_examples(train, k=12):
    """k labeled train rows, balanced across classes, fixed by the seed."""
    per = k // 2
    return pd.concat([train[train["label"] == c].sample(per, random_state=SEED) for c in (0, 1)]).sample(frac=1, random_state=SEED)


def evaluate_claude(run, system, train, val, test, few_shot=False):
    ensure_api_key()
    from concurrent.futures import ThreadPoolExecutor
    shots = fewshot_examples(train) if few_shot else None
    usage = {"input": 0, "output": 0}

    def predict(df):
        with ThreadPoolExecutor(max_workers=8) as ex:
            outs = list(ex.map(lambda r: claude_label(system, r, shots), [r for _, r in df.iterrows()]))
        for _, o in outs:
            for k in usage:
                usage[k] += o.get("_usage", {}).get(k, 0)
        return [lab for lab, _ in outs]

    val_pred = predict(val)
    test_pred = predict(test)
    report(run, {
        "val": scores(val["label"], val_pred),
        "settings": {"model": MODEL, "effort": EFFORT, "few_shot": len(shots) if few_shot else 0},
        "new_tokens_this_run": usage,
    })
    write_submission(run, test, test_pred)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "run1"
    train, val, test = load()
    print(f"train {len(train)}  val {len(val)}  test {len(test)}  train labels {train['label'].value_counts().to_dict()}")
    if run == "run1":
        evaluate_trainable(run, make_tfidf(), train, val, test, {"model": "tfidf(1-2gram, response)+logreg", "C": 1.0})
    elif run == "run2":
        cfg = {"hidden": 64, "epochs": 150, "lr": 1e-3, "weight_decay": 1e-2, "dropout": 0.3}
        evaluate_trainable(run, make_mlp(**cfg), train, val, test,
                           {"model": "MiniLM-L6 [context, response opening, response, context*opening] + MLP", **cfg})
    elif run == "run3":
        evaluate_claude(run, SHORT_PROMPT, train, val, test)
    elif run == "run4":
        evaluate_claude(run, GUIDELINES_PROMPT, train, val, test)
    elif run == "run5":
        evaluate_claude(run, GUIDELINES_PROMPT, train, val, test, few_shot=True)
    else:
        sys.exit(f"unknown run {run}")


if __name__ == "__main__":
    main()

"""ARI 410/510 Phase 2b - "Does the Reply Answer the Question?" (Roshkrishna Ranjith, Team 9)

Labels: 0 = Partially answered, 1 = Fully answered.  Metric: macro F1.

Usage (each command prints dev scores and writes submissions/<run>.csv):
    python reply_answers.py run1     # TF-IDF + logistic regression
    python reply_answers.py run2     # MiniLM sentence embeddings + small neural network (PyTorch)
    python reply_answers.py run3     # Claude zero-shot, short prompt
    python reply_answers.py run4     # Claude zero-shot, full annotation guidelines as the prompt
    python reply_answers.py run5     # Claude few-shot (labeled examples in the prompt) + guidelines

Dev evaluation: the labeled set is tiny (12 train + 4 validation), so every run is scored two ways:
  * "val"  - train on train.csv, score on the 4 validation.csv rows (what the host intended)
  * "loo"  - leave-one-out over all 16 labeled rows (more stable; each row predicted by a model
             that never saw it). LLM zero-shot runs have no training, so they are scored on all 16.
Final test predictions are made by a model trained on all 16 labeled rows.
The Claude runs need ANTHROPIC_API_KEY set in the environment.
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

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


def load():
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")
    return train, val, test


def split_text(text):
    """Split the 'TITLE ... QUESTION ... REPLY ...' text into (question, reply)."""
    if "\nREPLY\n" in text:
        q, r = text.split("\nREPLY\n", 1)
    else:
        q, r = text, ""
    return q.replace("TITLE\n", "").replace("\nQUESTION\n", "\n").strip(), r.strip()


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
# Trainable models (run1, run2): fit(train_df) -> predict(df) closures
# ---------------------------------------------------------------------------

def make_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    def fit(df):
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED),
        )
        model.fit(df["text"], df["label"])
        return lambda d: model.predict(d["text"])
    return fit


_EMB = {}


def embed(texts):
    """MiniLM sentence embeddings, cached in memory."""
    from sentence_transformers import SentenceTransformer
    if "model" not in _EMB:
        _EMB["model"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    missing = [t for t in texts if t not in _EMB]
    if missing:
        vecs = _EMB["model"].encode(missing, normalize_embeddings=True, show_progress_bar=False)
        for t, v in zip(missing, vecs):
            _EMB[t] = v
    return np.stack([_EMB[t] for t in texts])


def features(df):
    """[question, reply, |q - r|, q * r] built from separate question / reply embeddings."""
    qs, rs = zip(*(split_text(t) for t in df["text"]))
    q, r = embed(list(qs)), embed(list(rs))
    return np.hstack([q, r, np.abs(q - r), q * r]).astype(np.float32)


def make_mlp(hidden=32, epochs=200, lr=1e-3, weight_decay=1e-2, dropout=0.3):
    import torch
    from torch import nn

    def fit(df):
        set_seed()
        x = torch.tensor(features(df))
        y = torch.tensor(df["label"].values, dtype=torch.long)
        counts = torch.bincount(y, minlength=2).float().clamp(min=1)
        weights = counts.sum() / (2 * counts)          # class-balanced loss
        net = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
        net.eval()

        def predict(d):
            with torch.no_grad():
                return net(torch.tensor(features(d))).argmax(1).numpy()
        return predict
    return fit


def evaluate_trainable(run, fit, train, val, test, extra=None):
    set_seed()
    labeled = pd.concat([train, val], ignore_index=True)
    val_pred = fit(train)(val)
    loo_pred = []
    for i in range(len(labeled)):
        loo_pred.append(fit(labeled.drop(index=i))(labeled.iloc[[i]])[0])
    results = {"val": scores(val["label"], val_pred), "loo": scores(labeled["label"], loo_pred)}
    if extra:
        results["settings"] = extra
    report(run, results)
    write_submission(run, test, fit(labeled)(test))


# ---------------------------------------------------------------------------
# Claude runs (run3-run5)
# ---------------------------------------------------------------------------

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")

SHORT_PROMPT = """You judge whether a reply on a home-improvement Q&A site fully answers the question.

Labels:
- "Fully answered": the reply addresses every explicit request and important constraint in the question \
(directly, or by explaining why the requested result is not possible).
- "Partially answered": the reply addresses some of the request but leaves an important requested part \
or constraint unaddressed.

Judge request coverage only - not whether the advice is correct, safe, polite or popular. \
A brief reply can be complete, and a long reply can still miss the main question."""

SCHEMA = {
    "type": "object",
    "properties": {
        "requests": {"type": "array", "items": {"type": "string"},
                     "description": "The distinct explicit requests and material constraints in the question."},
        "reason": {"type": "string", "description": "What the reply covers and what, if anything, is missing."},
        "label": {"type": "string", "enum": ["Fully answered", "Partially answered"]},
    },
    "required": ["requests", "reason", "label"],
    "additionalProperties": False,
}

LABEL_TO_INT = {"Partially answered": 0, "Fully answered": 1}
INT_TO_LABEL = {0: "Partially answered", 1: "Fully answered"}


def guidelines_prompt():
    text = (DATA / "GUIDELINES.md").read_text(encoding="utf-8")
    # Keep the labeling rules; drop the annotation-logistics sections (tooling, contacts, calibration).
    text = text.split("## Issue flags and uncertainty")[0] + text.split("## Clarifications from the AI review")[-1]
    return ("You are an annotator for the task below. Follow these annotation guidelines exactly. "
            "No reference items use 'Did not answer', so choose between 'Fully answered' and "
            "'Partially answered' only; if a reply covers none of the requests, pick 'Partially answered'.\n\n"
            + text)


def item_block(text):
    q, r = split_text(text)
    return f"<question>\n{q}\n</question>\n\n<reply>\n{r}\n</reply>"


def shots_block(shot_df):
    parts = []
    for _, row in shot_df.iterrows():
        parts.append(f"<example>\n{item_block(row['text'])}\n<label>{INT_TO_LABEL[int(row['label'])]}</label>\n</example>")
    return "Labeled examples from the same dataset:\n\n" + "\n\n".join(parts)


def claude_label(system, item_text, shot_df=None, tag=""):
    """One Claude call -> (label_int, parsed_json). Responses are cached on disk so reruns are free."""
    import hashlib
    import anthropic

    user = ""
    if shot_df is not None and len(shot_df):
        user += shots_block(shot_df) + "\n\nNow label this item.\n\n"
    user += item_block(item_text)
    key = hashlib.sha256(f"{MODEL}|{system}|{user}".encode()).hexdigest()[:24]
    CACHE.mkdir(exist_ok=True)
    cpath = CACHE / f"{key}.json"
    if cpath.exists():
        out = json.loads(cpath.read_text())
    else:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=system,
            output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"refusal on {tag}")
        text = next(b.text for b in resp.content if b.type == "text")
        out = json.loads(text)
        out["_usage"] = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
        cpath.write_text(json.dumps(out, indent=2))
    return LABEL_TO_INT[out["label"]], out


def ensure_api_key():
    """Use ANTHROPIC_API_KEY from the environment; on Windows, fall back to the value saved with
    `setx` (stored under HKCU\\Environment) when the current process started before it was set."""
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
    sys.exit("ANTHROPIC_API_KEY is not set - see the setup notes.")


def evaluate_claude(run, system, train, val, test, few_shot=False):
    ensure_api_key()
    labeled = pd.concat([train, val], ignore_index=True)
    usage = {"input": 0, "output": 0}

    def predict(df, shots_for):
        preds = []
        for i, row in df.iterrows():
            label, out = claude_label(system, row["text"], shots_for(i, row), tag=row["id"])
            preds.append(label)
            for k in usage:
                usage[k] += out.get("_usage", {}).get(k, 0)
            print(f"  {row['id']}: {out['label']}")
        return preds

    if few_shot:
        # Dev: each labeled row is predicted with the OTHER labeled TRAIN rows as examples (never itself).
        val_pred = predict(val, lambda i, row: train)
        loo_pred = predict(labeled, lambda i, row: train[train["id"] != row["id"]])
        test_pred = predict(test, lambda i, row: labeled)
    else:
        val_pred = predict(val, lambda i, row: None)
        loo_pred = predict(labeled, lambda i, row: None)
        test_pred = predict(test, lambda i, row: None)

    report(run, {
        "val": scores(val["label"], val_pred),
        "all16": scores(labeled["label"], loo_pred),
        "settings": {"model": MODEL, "effort": "high", "few_shot": few_shot},
        "new_tokens_this_run": usage,
    })
    write_submission(run, test, test_pred)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "run1"
    train, val, test = load()
    print(f"train {len(train)}  val {len(val)}  test {len(test)}")
    print("label counts (train+val):", pd.concat([train, val])["label"].value_counts().to_dict())
    if run == "run1":
        evaluate_trainable(run, make_tfidf(), train, val, test, {"model": "tfidf(1-2gram)+logreg", "C": 1.0})
    elif run == "run2":
        cfg = {"hidden": 32, "epochs": 200, "lr": 1e-3, "weight_decay": 1e-2, "dropout": 0.3}
        evaluate_trainable(run, make_mlp(**cfg), train, val, test, {"model": "MiniLM-L6 [q,r,|q-r|,q*r] + MLP", **cfg})
    elif run == "run3":
        evaluate_claude(run, SHORT_PROMPT, train, val, test)
    elif run == "run4":
        evaluate_claude(run, guidelines_prompt(), train, val, test)
    elif run == "run5":
        evaluate_claude(run, guidelines_prompt(), train, val, test, few_shot=True)
    else:
        sys.exit(f"unknown run {run}")


if __name__ == "__main__":
    main()

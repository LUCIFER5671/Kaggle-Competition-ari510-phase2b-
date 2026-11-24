"""ARI 410/510 Phase 2b - Refactoring PR Consistency ("Does the PR Do What It Says?") (Roshkrishna Ranjith, Team 9)

Labels: Consistent, Inconsistent, Out of scope, Unavailable.

Usage (each command prints dev scores and writes submissions/<run>.csv):
    python pr_consistency.py run1   # rule for Unavailable + TF-IDF + logistic regression
    python pr_consistency.py run2   # rule for Unavailable + MiniLM embeddings (title+body, code) + MLP (PyTorch)
    python pr_consistency.py run3   # rule for Unavailable + Claude zero-shot with the label definitions
    python pr_consistency.py run4   # rule for Unavailable + Claude few-shot (labeled train PRs as examples)

Rule used by every run: a PR whose description AND code changes are both missing is "Unavailable"
(true for all 7 Unavailable rows in train and no other train row). The models only choose among the other 3 labels.

Dev evaluation: there is no validation file and only 31 labeled rows, so every run is scored with
leave-one-out (each row predicted by a model that never saw it). Claude zero-shot has no training, so it is
simply scored on all 31; few-shot uses the other labeled rows as examples. Test predictions use all 31 rows.
"""
import hashlib
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
LABELS = ["Consistent", "Inconsistent", "Out of scope", "Unavailable"]
MODEL_LABELS = ["Consistent", "Inconsistent", "Out of scope"]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load():
    train = pd.read_csv(DATA / "train.csv", dtype={"id": str})
    test = pd.read_csv(DATA / "test.csv", dtype={"id": str})
    for d in (train, test):
        d["unavailable"] = d["body"].isna() & d["code_changes"].isna()
        for c in ("title", "body", "code_changes"):
            d[c] = d[c].fillna("").astype(str)
    return train, test


def text_of(df):
    return (df["title"] + "\n" + df["body"] + "\n" + df["code_changes"]).tolist()


def scores(y_true, y_pred, labels=None):
    labels = labels or sorted(set(y_true))
    return {
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "n": len(y_true),
    }


def write_submission(run, test, preds):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{run}.csv"
    pd.DataFrame({"id": test["id"], "label": preds}).to_csv(path, index=False)
    print(f"wrote {path}  {pd.Series(preds).value_counts().to_dict()}")


def report(run, results):
    print(f"\n== {run} dev results ==")
    for k, v in results.items():
        print(f"  {k}: {v}")
    log = HERE / "results.json"
    allres = json.loads(log.read_text()) if log.exists() else {}
    allres[run] = results
    log.write_text(json.dumps(allres, indent=2))


def with_rule(model_predict):
    """Wrap a 3-class predictor: rows with no description and no code are 'Unavailable'."""
    def predict(df):
        out = np.array(["Unavailable"] * len(df), dtype=object)
        avail = ~df["unavailable"].values
        if avail.any():
            out[avail] = model_predict(df[avail])
        return list(out)
    return predict


def available(df):
    return df[~df["unavailable"]]


# ---------------------------------------------------------------------------
# Trainable models (fit on the available rows only)
# ---------------------------------------------------------------------------

def make_tfidf(C=1.0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    def fit(df):
        d = available(df)
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=20000),
            LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED),
        )
        model.fit(text_of(d), d["label"])
        return with_rule(lambda x: model.predict(text_of(x)))
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
    desc = embed((df["title"] + "\n" + df["body"]).tolist())
    code = embed(df["code_changes"].tolist())  # MiniLM reads the first ~256 tokens: the file list and first hunks
    return np.hstack([desc, code, desc * code]).astype(np.float32)


def make_mlp(hidden=32, epochs=150, lr=1e-3, weight_decay=1e-2, dropout=0.3):
    import torch
    from torch import nn

    def fit(df):
        set_seed()
        d = available(df)
        x = torch.tensor(features(d))
        y = torch.tensor([MODEL_LABELS.index(l) for l in d["label"]], dtype=torch.long)
        counts = torch.bincount(y, minlength=3).float().clamp(min=1)
        net = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 3))
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss(weight=counts.sum() / (3 * counts))
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss_fn(net(x), y).backward()
            opt.step()
        net.eval()

        def predict(xdf):
            with torch.no_grad():
                return [MODEL_LABELS[i] for i in net(torch.tensor(features(xdf))).argmax(1).numpy()]
        return with_rule(predict)
    return fit


def evaluate_trainable(run, fit, train, test, settings):
    set_seed()
    loo = [fit(train.drop(index=i))(train.loc[[i]])[0] for i in train.index]
    sub = available(train)
    report(run, {
        "loo": scores(train["label"], loo),
        "loo_available_rows_only": scores(sub["label"], [loo[train.index.get_loc(i)] for i in sub.index]),
        "settings": settings,
    })
    write_submission(run, test, fit(train)(test))


# ---------------------------------------------------------------------------
# Claude runs
# ---------------------------------------------------------------------------

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = "medium"
DEFINITIONS_FILE = HERE / "label_definitions.md"   # the host's label definitions, copied from the Kaggle Overview

SCHEMA = {
    "type": "object",
    "properties": {
        "claimed_changes": {"type": "array", "items": {"type": "string"},
                            "description": "What the PR title/description says the change does."},
        "observed_changes": {"type": "string", "description": "What the visible code changes actually do."},
        "reason": {"type": "string"},
        "label": {"type": "string", "enum": MODEL_LABELS},
    },
    "required": ["claimed_changes", "observed_changes", "reason", "label"],
    "additionalProperties": False,
}


def system_prompt():
    return ("You label GitHub pull requests that claim to be refactorings. Compare what the title and description "
            "say with what the code changes actually do, then choose one label. The code diff may be truncated; "
            "judge from what is shown.\n\nLabel definitions from the dataset:\n\n"
            + DEFINITIONS_FILE.read_text(encoding="utf-8"))


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


def item_block(row, code_chars=6000):
    return (f"<title>{row['title']}</title>\n<description>\n{row['body']}\n</description>\n"
            f"<code_changes>\n{row['code_changes'][:code_chars]}\n</code_changes>")


def shots_block(shots):
    parts = [f"<example>\n{item_block(r, code_chars=1500)}\n<label>{r['label']}</label>\n</example>"
             for _, r in shots.iterrows()]
    return "Labeled examples from the same dataset (code shortened):\n\n" + "\n\n".join(parts) + "\n\nNow label this PR.\n\n"


def claude_label(system, row, shots=None):
    import anthropic
    user = (shots_block(shots) if shots is not None else "") + item_block(row)
    key = hashlib.sha256(f"{MODEL}|{EFFORT}|{system}|{user}".encode()).hexdigest()[:24]
    CACHE.mkdir(exist_ok=True)
    cpath = CACHE / f"{key}.json"
    if cpath.exists():
        out = json.loads(cpath.read_text())
    else:
        resp = anthropic.Anthropic().messages.create(
            model=MODEL, max_tokens=4000, system=system,
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"refusal on {row['id']}")
        out = json.loads(next(b.text for b in resp.content if b.type == "text"))
        out["_usage"] = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
        cpath.write_text(json.dumps(out, indent=2))
    return out["label"], out


def evaluate_claude(run, train, test, few_shot=False):
    ensure_api_key()
    from concurrent.futures import ThreadPoolExecutor
    system = system_prompt()
    usage = {"input": 0, "output": 0}
    labeled_avail = available(train)

    def shots_for(row, pool):
        if not few_shot:
            return None
        return pool[pool["id"] != row["id"]]

    def predict(df, pool):
        rows = [r for _, r in df.iterrows() if not r["unavailable"]]
        with ThreadPoolExecutor(max_workers=8) as ex:
            outs = list(ex.map(lambda r: claude_label(system, r, shots_for(r, pool)), rows))
        for _, o in outs:
            for k in usage:
                usage[k] += o.get("_usage", {}).get(k, 0)
        it = iter(lab for lab, _ in outs)
        return ["Unavailable" if r["unavailable"] else next(it) for _, r in df.iterrows()]

    dev = predict(train, labeled_avail)
    sub = available(train)
    report(run, {
        "all31": scores(train["label"], dev),
        "available_rows_only": scores(sub["label"], [dev[train.index.get_loc(i)] for i in sub.index]),
        "settings": {"model": MODEL, "effort": EFFORT, "few_shot": few_shot},
        "new_tokens_this_run": usage,
    })
    write_submission(run, test, predict(test, labeled_avail))


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "run1"
    train, test = load()
    print(f"train {len(train)}  test {len(test)}  labels {train['label'].value_counts().to_dict()}  "
          f"test unavailable {int(test['unavailable'].sum())}")
    if run == "run1":
        evaluate_trainable(run, make_tfidf(), train, test, {"model": "rule + tfidf(1-2gram)+logreg", "C": 1.0})
    elif run == "run2":
        cfg = {"hidden": 32, "epochs": 150, "lr": 1e-3, "weight_decay": 1e-2, "dropout": 0.3}
        evaluate_trainable(run, make_mlp(**cfg), train, test,
                           {"model": "rule + MiniLM-L6 [desc, code, desc*code] + MLP", **cfg})
    elif run == "run3":
        evaluate_claude(run, train, test)
    elif run == "run4":
        evaluate_claude(run, train, test, few_shot=True)
    else:
        sys.exit(f"unknown run {run}")


if __name__ == "__main__":
    main()

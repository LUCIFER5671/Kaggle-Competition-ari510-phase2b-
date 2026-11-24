"""ARI 410/510 Phase 2b - What Makes a Title Interesting? (Roshkrishna Ranjith, Team 9)

Regression: predict a 1-6 interest score from a title and its domain. Metric: MAE (lower is better).

Key data fact: train labels come from ONE annotator per title; test labels are the MEAN of 6 annotators.
Single ratings are very noisy (spread almost uniformly over 1-6), while a 6-annotator mean sits close to the middle.
The best prediction for a mean of annotators is the expected rating E[y | title], so every model here is fit to
predict that expectation (squared-error regression with regularization, or a linear calibration of an LLM score),
which automatically shrinks predictions toward the center by an amount learned from the data.

Usage (each command prints dev scores and writes submissions/<run>.csv):
    python title_interest.py run1   # TF-IDF (word + char n-grams) + domain -> ridge regression
    python title_interest.py run2   # MiniLM title embedding + domain -> MLP regressor (PyTorch)
    python title_interest.py run3   # Claude zero-shot 1-6 rating, linearly calibrated on train labels
    python title_interest.py run4   # ridge stack: Claude rating + domain + MiniLM embedding

Dev evaluation: 10-fold cross-validation (repeated 3x with different splits) on the 120 train titles.
  * mae_single  - MAE against the single-annotator train labels. This is NOT the test metric's scale: a single rating
                  is far noisier than a 6-annotator mean, so every model scores ~1.2-1.5 here. Use it to compare runs.
  * spearman    - rank correlation with the train labels: how well a model orders titles by interest. This transfers
                  directly to the test set (the ordering of expected interest is the same).
  * const_mae   - the same MAE for always predicting the train mean, for reference.
"""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

SEED = 42
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "submissions"
CACHE = HERE / "cache"
DOMAINS = ["article", "podcast", "reddit", "youtube"]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load():
    return pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")


def domain_onehot(df):
    return np.stack([(df["domain"] == d).astype(float).values for d in DOMAINS], axis=1)


def clip(p):
    return np.clip(p, 1.0, 6.0)


def cv_eval(fit, train, repeats=3):
    """Repeated 10-fold CV. fit(train_df) -> predict(df) -> array of scores."""
    y = train["interest"].values.astype(float)
    maes, rhos, const = [], [], []
    for rep in range(repeats):
        pred = np.zeros(len(train))
        const_pred = np.zeros(len(train))
        for tr, te in KFold(10, shuffle=True, random_state=SEED + rep).split(train):
            pred[te] = fit(train.iloc[tr])(train.iloc[te])
            const_pred[te] = y[tr].mean()
        maes.append(np.abs(pred - y).mean())
        rhos.append(spearmanr(pred, y).correlation)
        const.append(np.abs(const_pred - y).mean())
    return {"mae_single": round(float(np.mean(maes)), 4), "spearman": round(float(np.mean(rhos)), 4),
            "const_mae": round(float(np.mean(const)), 4)}


def write_submission(run, test, preds):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{run}.csv"
    preds = np.round(clip(np.asarray(preds, dtype=float)), 3)
    pd.DataFrame({"title_id": test["title_id"], "interest": preds}).to_csv(path, index=False)
    print(f"wrote {path}  mean {preds.mean():.2f}  min {preds.min():.2f}  max {preds.max():.2f}  sd {preds.std():.2f}")


def report(run, results):
    print(f"\n== {run} dev results ==")
    for k, v in results.items():
        print(f"  {k}: {v}")
    log = HERE / "results.json"
    allres = json.loads(log.read_text()) if log.exists() else {}
    allres[run] = results
    log.write_text(json.dumps(allres, indent=2))


def run_model(run, fit, train, test, settings):
    set_seed()
    res = cv_eval(fit, train)
    res["settings"] = settings
    report(run, res)
    write_submission(run, test, fit(train)(test))


# ---------------------------------------------------------------------------
# run1: TF-IDF + ridge
# ---------------------------------------------------------------------------

def make_tfidf_ridge(alpha=3.0):
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    def fit(df):
        word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, lowercase=False)
        x = hstack([word.fit_transform(df["title"]), char.fit_transform(df["title"]), csr_matrix(domain_onehot(df))])
        model = Ridge(alpha=alpha).fit(x, df["interest"])

        def predict(d):
            xd = hstack([word.transform(d["title"]), char.transform(d["title"]), csr_matrix(domain_onehot(d))])
            return clip(model.predict(xd))
        return predict
    return fit


# ---------------------------------------------------------------------------
# run2: MiniLM embedding + MLP regressor
# ---------------------------------------------------------------------------

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


def make_mlp(hidden=32, epochs=30, lr=1e-3, weight_decay=1e-1, dropout=0.3):
    import torch
    from torch import nn

    def fit(df):
        set_seed()
        x = torch.tensor(np.hstack([embed(list(df["title"])), domain_onehot(df)]), dtype=torch.float32)
        mu = float(df["interest"].mean())
        y = torch.tensor(df["interest"].values - mu, dtype=torch.float32)   # predict deviation from the mean
        net = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            nn.functional.mse_loss(net(x).squeeze(1), y).backward()
            opt.step()
        net.eval()

        def predict(d):
            xd = torch.tensor(np.hstack([embed(list(d["title"])), domain_onehot(d)]), dtype=torch.float32)
            with torch.no_grad():
                return clip(net(xd).squeeze(1).numpy() + mu)
        return predict
    return fit


# ---------------------------------------------------------------------------
# run3 / run4: Claude rating
# ---------------------------------------------------------------------------

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = "medium"

SYSTEM = """You estimate how interesting a content title is to a typical reader browsing online.

You will see a title and where it comes from (YouTube, Reddit, a podcast, or an article). Rate how interesting the
title makes the content seem, from 1 (not interesting: I would scroll past) to 6 (very interesting: I would click).
Judge the title itself, as a general reader would, not your own expertise in the topic.

Things that tend to affect perceived interest (the dataset's annotators also noted these):
- curiosity gap: the title hints at something without giving it away
- emotional language or exaggeration (can attract, or feel like clickbait)
- direct address to the reader ("you")
- missing details: unclear what the content is about, or jargon only a niche audience understands

Give your best estimate of the AVERAGE rating a group of ordinary readers would give. Decimals are fine."""

SCHEMA = {
    "type": "object",
    "properties": {
        "observations": {"type": "string", "description": "One or two sentences on what makes the title more or less interesting."},
        "score": {"type": "number", "description": "Expected average interest rating, 1.0 to 6.0."},
    },
    "required": ["observations", "score"],
    "additionalProperties": False,
}


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


def claude_score(row):
    """Raw Claude rating for one title. Cached on disk, so reruns are free and identical."""
    import anthropic
    user = f"Source: {row['domain']}\nTitle: {row['title']}"
    key = hashlib.sha256(f"{MODEL}|{EFFORT}|{SYSTEM}|{user}".encode()).hexdigest()[:24]
    CACHE.mkdir(exist_ok=True)
    cpath = CACHE / f"{key}.json"
    if cpath.exists():
        return json.loads(cpath.read_text())["score"]
    resp = anthropic.Anthropic().messages.create(
        model=MODEL, max_tokens=2000, system=SYSTEM,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refusal on {row['title_id']}")
    out = json.loads(next(b.text for b in resp.content if b.type == "text"))
    out["_usage"] = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
    cpath.write_text(json.dumps(out, indent=2))
    return out["score"]


def add_claude_scores(*dfs):
    ensure_api_key()
    from concurrent.futures import ThreadPoolExecutor
    for d in dfs:
        with ThreadPoolExecutor(max_workers=8) as ex:
            d["claude"] = list(ex.map(claude_score, [r for _, r in d.iterrows()]))


def make_calibrated_claude():
    """Linear map from Claude's raw score to the expected annotator rating, fit by least squares on the train fold.
    The fitted slope is < 1 when ratings are noisy, which pulls predictions toward the mean."""
    from sklearn.linear_model import LinearRegression

    def fit(df):
        lr = LinearRegression().fit(df[["claude"]], df["interest"])
        return lambda d: clip(lr.predict(d[["claude"]]))
    return fit


def make_stack(alpha=10.0):
    """Ridge on [Claude score, domain one-hot, MiniLM embedding]."""
    from sklearn.linear_model import Ridge

    def feats(d):
        return np.hstack([d[["claude"]].values, domain_onehot(d), embed(list(d["title"]))])

    def fit(df):
        model = Ridge(alpha=alpha).fit(feats(df), df["interest"])
        return lambda d: clip(model.predict(feats(d)))
    return fit


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "run1"
    train, test = load()
    print(f"train {len(train)}  test {len(test)}  train mean {train['interest'].mean():.3f}")
    if run == "run1":
        run_model(run, make_tfidf_ridge(), train, test, {"model": "tfidf word(1-2)+char(2-4) + domain -> ridge", "alpha": 3.0})
    elif run == "run2":
        cfg = {"hidden": 32, "epochs": 30, "lr": 1e-3, "weight_decay": 1e-1, "dropout": 0.3}
        run_model(run, make_mlp(**cfg), train, test, {"model": "MiniLM-L6 + domain -> MLP regressor (MSE)", **cfg})
    elif run in ("run3", "run4"):
        add_claude_scores(train, test)
        raw_rho = spearmanr(train["claude"], train["interest"]).correlation
        print(f"raw Claude score vs train labels: spearman {raw_rho:.3f}, raw MAE {np.abs(train['claude'] - train['interest']).mean():.3f}")
        if run == "run3":
            run_model(run, make_calibrated_claude(), train, test,
                      {"model": f"{MODEL} zero-shot score -> linear calibration", "effort": EFFORT, "raw_spearman": round(float(raw_rho), 4)})
        else:
            run_model(run, make_stack(), train, test,
                      {"model": "ridge on [Claude score, domain, MiniLM embedding]", "alpha": 10.0})
    else:
        sys.exit(f"unknown run {run}")


if __name__ == "__main__":
    main()

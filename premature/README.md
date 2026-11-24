# Detecting Premature Problem Solving — ARI 410/510 Phase 2b

Roshkrishna Ranjith (Team 9). Code for every run submitted to the Kaggle competition.
Experiment log and discussion: [NOTES.md](NOTES.md).

## Setup
1. Python 3.11, then `pip install -r requirements.txt`
   (CPU PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`).
2. Download the competition data from Kaggle and unzip it into `data/` (`train.csv`, `validation.csv`, `test.csv`).
   The data is not in this repo.
3. Claude runs only: set `ANTHROPIC_API_KEY` in your environment.

## Reproduce each run
Seeds are fixed (42). Each command prints the dev scores and writes `submissions/<run>.csv`.

| Run ID | Command | Approach |
|---|---|---|
| run1 | `python premature.py run1` | TF-IDF (1–2 grams) + logistic regression |
| run2 | `python premature.py run2` | MiniLM-L6 embeddings + MLP (PyTorch) |
| run3 | `python premature.py run3` | Claude Opus 5.5 zero-shot, short prompt |
| run4 | `python premature.py run4` | Claude Opus 5.5 zero-shot, host's annotation rules + examples |
| run5 | `python premature.py run5` | Claude Opus 5.5 few-shot (12 examples) + rules |

Every Claude response used for the submitted runs is stored in `cache/`. With the cache present, runs 3–5 reproduce the
submitted files exactly and make no API calls.

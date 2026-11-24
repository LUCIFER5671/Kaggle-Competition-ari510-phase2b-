# Does the PR Do What It Says? — ARI 410/510 Phase 2b

Roshkrishna Ranjith (Team 9). Code for every run submitted to the Kaggle competition.
Experiment log and discussion: [NOTES.md](NOTES.md).

## Setup
1. Python 3.11, then `pip install -r requirements.txt`
   (CPU PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`).
2. Download the competition data from Kaggle and unzip it into `data/` (`train.csv`, `test.csv`). Not in this repo.
3. Claude runs only: set `ANTHROPIC_API_KEY` in your environment.

## Reproduce each run
Seeds are fixed (42). Each command prints the dev scores and writes `submissions/<run>.csv`.

| Run ID | Command | Approach |
|---|---|---|
| run1 | `python pr_consistency.py run1` | Unavailable rule + TF-IDF + logistic regression |
| run2 | `python pr_consistency.py run2` | Unavailable rule + MiniLM-L6 embeddings + MLP (PyTorch) |
| run3 | `python pr_consistency.py run3` | Unavailable rule + Claude Opus 5.5 zero-shot with label definitions |
| run4 | `python pr_consistency.py run4` | Unavailable rule + Claude Opus 5.5 few-shot |

The Claude prompt uses `label_definitions.md` (from the competition Overview). Every Claude response used for the
submitted runs is stored in `cache/`; with it present, runs 3–4 reproduce the submitted files exactly with no API calls.

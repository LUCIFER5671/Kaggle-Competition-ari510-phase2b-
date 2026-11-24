# What Makes a Title Interesting? — ARI 410/510 Phase 2b

Roshkrishna Ranjith (Team 9). Code for every run submitted to the Kaggle competition.
Experiment log and discussion: [NOTES.md](NOTES.md).

## Setup
1. Python 3.11, then `pip install -r requirements.txt`
   (CPU PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`).
2. Download the competition data from Kaggle and unzip it into `data/` (`train.csv`, `test.csv`). Not in this repo.
3. Claude runs only: set `ANTHROPIC_API_KEY` in your environment.

## Reproduce each run
Seeds are fixed (42). Each command prints the cross-validation scores and writes `submissions/<run>.csv`.

| Run ID | Command | Approach |
|---|---|---|
| run1 | `python title_interest.py run1` | TF-IDF + domain → ridge regression |
| run2 | `python title_interest.py run2` | MiniLM-L6 embedding + domain → MLP regressor (PyTorch) |
| run3 | `python title_interest.py run3` | Claude Opus 5.5 zero-shot rating → linear calibration |
| run4 | `python title_interest.py run4` | Ridge on [Claude rating, domain, MiniLM embedding] |

Every Claude response used for the submitted runs is stored in `cache/`; with it present, runs 3–4 reproduce the
submitted files exactly with no API calls.

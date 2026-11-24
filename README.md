# ARI 410/510 Phase 2b — Kaggle competition entries

Roshkrishna Ranjith (Team 9). Code for every run I submitted to four classmates' Kaggle competitions.
Each folder has its own README (how to reproduce each run ID) and NOTES.md (experiments, dev and leaderboard scores).

| Folder | Competition | Runs |
|---|---|---|
| [reply-answers](reply-answers/) | Does the Reply Answer the Question? (assigned) | run1–run5 |
| [premature](premature/) | Detecting Premature Problem Solving in Mental Health Counseling Responses | run1–run5 |
| [pr-consistency](pr-consistency/) | Does the PR Do What It Says? (refactoring PR consistency) | run1–run4 |
| [title-interest](title-interest/) | What Makes a Title Interesting? (regression, MAE) | run1–run4 |

## Setup (all three)
```
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Mac/Linux)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r reply-answers/requirements.txt
```
Download each competition's data from Kaggle into that folder's `data/` (data is not redistributed here).
Claude runs need `ANTHROPIC_API_KEY`; with the committed `cache/` folders they reproduce the submitted files with no API calls.

Approaches in every competition: a TF-IDF + logistic regression baseline, a MiniLM sentence-embedding + MLP neural
network trained with PyTorch, and Claude (Opus 5.5) zero-shot / few-shot prompting called through the Anthropic API.
All seeds are fixed (42).

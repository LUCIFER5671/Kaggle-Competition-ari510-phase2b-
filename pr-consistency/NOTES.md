# Experiment notes – Does the PR Do What It Says? (Refactoring PR consistency)

Task: given an agent-authored refactoring PR's title, description and code diff, classify it as Consistent, Inconsistent,
Out of scope (not really an Extract/Move refactoring) or Unavailable (PR no longer accessible).
Data: 31 labeled train rows (14 Consistent, 7 Unavailable, 5 Out of scope, 5 Inconsistent), 14 test rows, no validation
file. Human agreement was low (Krippendorff's α = 0.37), so labels are noisy.
Metric: macro F1 over the 4 classes. Host baselines (public LB): TF-IDF + logistic regression 0.375, random 0.375,
majority class 0.136. Public leaderboard ≈ 57% of the 14 test rows (≈8 rows). Best public: run3 0.881, 1st on the public
leaderboard (next competitor 0.560). Final selection: run3 + run4 (the two best runs on dev).
run3 beat run4 on the public rows although run4 was better in leave-one-out (0.84 vs 0.77); on ~8 rows that is one or
two predictions, so I kept both as finals rather than trusting either score alone. run2 (0.375) only tied the host's
baselines on the public rows despite 0.63 in leave-one-out.

Dev evaluation: leave-one-out over all 31 rows (each row predicted by a model that never saw it). I also report the score
on the 24 rows that are not Unavailable, since those are the ones that need a model.

| Run | Approach | LOO macro F1 (31) | Non-Unavailable rows (24) | Public LB |
|---|---|---|---|---|
| run1 | Unavailable rule + TF-IDF (1–2 grams, title+description+diff) + logistic regression, class-balanced | 0.434 | 0.246 | 0.417 |
| run2 | Unavailable rule + MiniLM-L6 embeddings of (title+description) and diff → [desc, code, desc×code] → MLP (hidden 32, dropout 0.3, AdamW lr 1e-3, wd 1e-2, 150 epochs, class-weighted loss) | 0.628 | 0.504 | 0.375 |
| run3 | Unavailable rule + Claude Opus 5.5 zero-shot with the host's label definitions; JSON output (claimed changes → observed changes → reason → label); effort medium | 0.774 | 0.698 | **0.881** |
| run4 | run3 + few-shot: every other labeled PR as an example (diff shortened to 1,500 chars per example) | **0.841** | **0.788** | 0.750 |

API cost: run3 ≈ $0.50, run4 ≈ $2.80 (674k input tokens: 23 examples per call). Responses cached in `cache/`, so reruns are
free and identical.

## Preprocessing
- **Unavailable rule:** all 7 Unavailable train rows are exactly the rows whose description AND code changes are missing
  (no other row has both missing). Every run labels such rows Unavailable in code; 3 test rows match. Models only choose
  among the other 3 labels. This follows directly from the label definition (the PR can't be accessed, so its text and
  diff could not be collected).
- Missing single fields filled with empty strings. Diffs are already truncated to 6,000 chars by the host.
- run2 embeds the description and the diff separately so the classifier can compare them (desc×code term). MiniLM only
  reads the first ~256 tokens of each, so it mostly sees the changed-file list and the first hunks.

## Prompt choices (LLM)
- The label definitions are copied from the host's Overview into `label_definitions.md` and used as the system prompt.
- The JSON output makes the model first list what the PR claims, then summarize what the visible diff does, before the
  label. This mirrors the task: compare claim vs code.
- The prompt says the diff may be truncated and to judge from what is shown, so it does not call a PR Inconsistent just
  because part of the code is cut off.
- Few-shot (run4) helped here (+0.07 on the 31 rows), unlike my other two competitions. The boundaries (especially
  "Out of scope" vs a real Extract/Move refactoring) are dataset-specific and hard to convey from one-line definitions, so
  seeing labeled PRs helps. In dev, each PR's examples exclude itself.

## Not submitted
- TF-IDF C ∈ {0.3, 1, 10}: all 0.434 LOO. Text n-grams from 24 non-trivial rows can't separate 3 classes; it predicts
  mostly Consistent.
- MLP hidden ∈ {16, 32, 64} and 150 vs 400 epochs: all 0.628 LOO. Kept the original settings.

## What worked
Claude few-shot (run4). The label depends on comparing two things (what the description claims vs what the diff does),
which needs reading code. With 24 usable training rows, the trained models can't learn that comparison; the embedding MLP
beat TF-IDF because the separate description/diff embeddings at least capture topic overlap. The rule for Unavailable is
the single biggest win for every run: it gets an entire class (a quarter of the macro F1) right from a data property.

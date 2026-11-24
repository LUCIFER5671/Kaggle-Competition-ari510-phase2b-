# Experiment notes – Does the Reply Answer the Question?

Metric: macro F1 (0 = Partially answered, 1 = Fully answered). Host baselines: dummy 0.333, trained 0.333 (public LB).
Labeled data: 16 rows (10 Fully, 6 Partially). Reference point: always predicting "Fully" = macro F1 0.385 on the 16.

| Run | Approach | val (4) macro F1 | LOO / all-16 macro F1 | Public LB |
|---|---|---|---|---|
| run1 | TF-IDF (1–2 grams) + logistic regression, class-balanced, C=1 | 0.429 | 0.360 | 0.333 |
| run2 | MiniLM-L6 embeddings of question and reply → [q, r, \|q−r\|, q·r] → MLP (hidden 32, dropout 0.3, AdamW lr 1e-3, wd 1e-2, 200 epochs, class-weighted loss) | 0.333 | 0.377 | 0.500 |
| run3 | Claude Opus 5.5 zero-shot, short label-definition prompt, JSON output (list requests → reason → label), effort high | 1.000 | 0.867 | 0.500 |
| run4 | Same as run3 but the prompt is the full annotation guidelines (labeling rules + tricky-case rules + clarifications) | 1.000 | 0.867 | 0.500 |
| run5 | run4 prompt + few-shot: the labeled train rows as examples (dev rows never see their own label; test uses all 16) | 1.000 | 0.867 | 0.500 |

LLM runs are scored on all 16 labeled rows (no training involved for zero-shot; for few-shot, each dev row's examples exclude itself).
API cost for runs 3–5 together: about $2.40 (≈427k input + 36k output tokens on Opus 5.5). Responses are cached in `cache/`, so reruns are free and give identical output.

Public leaderboard = ~50% of the 8 test rows (about 4 rows), so one row changes the score by a lot.
Runs 2–5 all scored 0.500 public; the top public score is 0.733 (3 of 4). Final selection: run3 + run5 (best dev score, and they
disagree on one test row). I did not tune anything to the public score - the handout warns against it, and 4 rows is pure noise.

## Prompt choices (LLM)
- Structured JSON output with three fields in order: the question's distinct requests, a reason, then the label. Making the model list requests first mirrors step 2 of the annotation guidelines ("make a short list of distinct requests and constraints").
- Prompt A (run3): short definitions of the two labels + "judge coverage, not correctness".
- Prompt B (run4): the actual annotator guidelines, minus tooling/contact sections. No dev gain over A, ~3.5× the input tokens.
- Prompt C (run5): B + labeled examples. No dev gain, ~3× more tokens than B; changed one test prediction vs B.
- "Did not answer" is not in the reference labels, so the prompt restricts output to the two classes.

## Not submitted
- run2 hyperparameter sweep (LOO macro F1): hidden 16 → 0.417; hidden 64 → 0.377; epochs 50 vs 200 and weight decay 0.01 vs 0.1 made no difference.
  All within noise of the always-"Fully" reference (0.385), so I kept the original settings rather than pick a winner on 16 rows.

## What worked
Claude (runs 3–5) at 0.867 dev macro F1 vs ~0.37 for the trained models and 0.333 for the host baselines.
With 12–16 training rows, bag-of-words and embedding classifiers can't learn "does the reply cover every request"; they mostly pick up topic words
(the embedding of a reply is similar to its question whether or not it answers it). The task needs reasoning: list what was asked, then check each item
against the reply. An LLM can do that without training data, which is why it wins here. Adding the full guidelines or examples didn't change the dev score;
on 16 rows the three prompts can't be told apart, so the short prompt (run3) is the most cost-effective.

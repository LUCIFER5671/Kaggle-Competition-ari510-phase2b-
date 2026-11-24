# Experiment notes – Detecting Premature Problem Solving

Task: given a person's mental-health message (context) and a counselor's reply (response), predict whether the reply
moves into advice/solutions before adequately acknowledging the emotional concern (1 = premature, 0 = acknowledges first).
Data: 120 train (74 premature / 46 not), 30 validation, 60 test. Host's annotators had 76% average pairwise agreement in
calibration (11/20 unanimous), so labels are noisy and no model should be expected to approach 100%.
Metric reported here: macro F1. Validation has only 30 rows, so a difference of ~0.05 is about one or two rows.
Host baselines (public LB): majority class 0.388, sample_submission 0.268. Best public score: run4 0.641 (2nd on the public
leaderboard, behind a host-team member's 0.691). Final selection: run4 + run5 (best two on validation).
The three Claude runs keep the same order on public as on validation (run4 > run5 > run3). TF-IDF fell from 0.691 (val) to
0.498 (public) and below run3, consistent with its lower 5-fold CV score (0.593): its validation score was a lucky split.

| Run | Approach | val (30) macro F1 | 5-fold CV (150) macro F1 | Public LB |
|---|---|---|---|---|
| run1 | TF-IDF (1–2 grams) on the response + logistic regression, class-balanced, C=1 | 0.691 | 0.593 | 0.498 |
| run2 | MiniLM-L6 embeddings of context, response opening (first 2 sentences), full response, context×opening → MLP (hidden 64, dropout 0.3, AdamW lr 1e-3, wd 1e-2, 150 epochs, class-weighted loss) | 0.554 | 0.547 | 0.475 |
| run3 | Claude Opus 5.5 zero-shot, short label definition; JSON output (concern → first advice → what acknowledged it before → label); effort medium | 0.653 | – | 0.574 |
| run4 | Claude Opus 5.5 zero-shot, host's annotation rule table + their two examples + a 4-step procedure (find first advice, check what precedes it) | **0.832** | – | **0.641** |
| run5 | run4 prompt + 12 labeled train examples (6 per class, fixed seed) | 0.733 | – | 0.630 |

Claude runs are scored on validation only (each labeled row would cost an API call). API cost for runs 3–5: about $4.60
(≈776k input + 77k output tokens). Responses are cached in `cache/`, so reruns are free and identical.

## Preprocessing
- Normalized whitespace and non-breaking spaces; replaced the broken "�" quote characters in the source text.
- The decision depends on what comes first in the reply, so run2 embeds the response opening (first 2 sentences)
  separately from the full response, and run4's prompt asks the model to locate the first piece of advice.

## Prompt development (LLM)
- run3 (short definition) over-predicted "premature": 23 of 30 predicted premature vs 18 true; 7 of 12 "acknowledges
  first" replies were called premature. Claude was stricter than the annotators about what counts as acknowledgement.
- For run4 I used the host's own rule table and examples, and stated that acknowledgement can be brief. An earlier draft
  of this prompt included my own extra rule ("a bare thanks is not adequate acknowledgement"); I removed it before running
  because it was not in the host's guidelines and would make the model even stricter. run4 = 0.832 on validation.
  Caveat: this change was motivated by run3's validation errors, so the validation score is slightly optimistic.
- run5 added 12 balanced labeled examples to the run4 prompt; it did worse (0.733) at ~4× the input tokens.
  With noisy labels, a handful of examples may teach the wrong boundary as easily as the right one.

## Not submitted
- TF-IDF C sweep (5-fold CV): C=0.3 → 0.593, 3 → 0.605, 10 → 0.580. Kept C=1 (differences are within noise).
- MLP sweep (5-fold CV): hidden 32 / 100 epochs → 0.561, hidden 64 / 300 epochs → 0.556, hidden 128 / 150 epochs → 0.561.
  Kept the original settings.

## What worked
Claude zero-shot with the host's guidelines (run4). Trained models only see 120 examples and pick up surface words
("suggest", "try", "sorry"), but the label depends on order (does acknowledgement come *before* the first advice?) and on a
judgment of "adequate", which an LLM can apply directly from the written rules. TF-IDF still beat the embedding MLP: sentence
embeddings capture topic (anxiety, anger, breakups) more than the advice-vs-empathy style that decides the label.

# Experiment notes – What Makes a Title Interesting?

Task: predict how interesting a content title is (1–6) from the title and its domain (YouTube, Reddit, podcast, article).
Metric: mean absolute error (lower is better).
Data: 120 train titles, each rated by ONE annotator (ratings spread almost evenly: 30×1, 20×2, 15×3, 25×4, 19×5, 11×6);
30 test titles whose label is the MEAN of 6 annotators.

## The key decision: predict the expected rating, shrunk toward the middle
A single rating is extremely noisy; an average of 6 ratings is not, and it sits near the middle of the scale. The best
prediction for an average of annotators is the expected rating E[rating | title], not a confident 1 or 6. So every model
is a regularized squared-error regression on the train labels, or (for Claude) a linear calibration of Claude's score
fit on the train labels. With noisy labels the fitted slope is small, which pulls predictions toward the mean by an
amount learned from the data rather than picked by hand.

## Dev evaluation
Repeated 10-fold cross-validation (3 repeats) on the 120 train titles.
- `mae_single`: MAE against single-annotator labels. Not on the test metric's scale (a single rating is much noisier than
  a 6-annotator mean, so everything scores 1.3–1.5), but it compares runs fairly.
- `spearman`: rank correlation with the labels: how well a model orders titles. This is the part that transfers to test.
- Reference: always predicting the training mean gets mae_single 1.485.

| Run | Approach | CV mae_single | CV Spearman | Test prediction spread (sd) | Public LB |
|---|---|---|---|---|---|
| run1 | TF-IDF word (1–2) + char (2–4) n-grams + domain → ridge (α=3) | 1.461 | 0.079 | 0.40 | |
| run2 | MiniLM-L6 title embedding + domain → MLP regressor (hidden 32, dropout 0.3, AdamW lr 1e-3, wd 0.1, 30 epochs, MSE on deviation from the mean) | 1.475 | 0.026 | 0.11 | |
| run3 | Claude Opus 5.5 zero-shot rating (1–6, decimals) → linear calibration fit on train labels | **1.385** | **0.309** | 0.44 | |
| run4 | Ridge (α=10) on [Claude rating, domain, MiniLM embedding] | 1.390 | 0.312 | 0.37 | |

Claude API cost: about $0.60 (150 calls). Responses cached in `cache/`, so reruns are free and identical.

## Prompt (run3)
- System prompt: rate how interesting the title makes the content seem to a typical reader, 1 (scroll past) to 6
  (would click); judge as a general reader, not by topic expertise; give the expected AVERAGE rating of a group, decimals
  allowed. It lists the characteristics the host's annotators recorded: curiosity gap, emotional language /
  exaggeration, direct address, missing details.
- Structured JSON output: short observations, then the numeric score.
- Raw Claude scores already rank titles reasonably (Spearman 0.33 with the single labels) but are spread too widely for
  the averaged test labels (raw MAE vs single labels 1.37). The linear calibration keeps the ordering and fixes the scale.

## Not submitted
- MLP settings (CV): 300 epochs → mae 1.573 (overfits: test predictions spread from 1.7 to 5.8); 100 epochs → 1.502;
  hidden 16 / 50 epochs → 1.479; hidden 16 / wd 1.0 → 1.488; 30 epochs → 1.475 (kept: least overfit). None has a
  meaningful rank correlation; the network learns little from 120 noisy labels beyond the mean.
- Ridge α ∈ {1, 3, 10, 30}: mae 1.468–1.472, Spearman 0.02–0.08. Kept α=3.

## What worked
Claude's rating, calibrated on the training labels (run3). Whether a title is interesting depends on world knowledge
(is "RuneScape 4th MMO teaser" exciting? is "Bougie Hoes Pt 2" a known artist?) and on reading style cues like curiosity
gaps, which 120 noisy labels can't teach a model from scratch. The trained models' rank correlation is near zero, so
they mostly predict the mean. Calibration matters as much as the LLM: an uncalibrated 1–6 score would be far too
extreme for labels that are averages of 6 people.

# Leaderboard probe submissions (diagnostic only)

Each file is a full, validator-passing `matching_results.tsv` (gzipped). `gunzip -k <file>` before uploading.
They change exactly one thing relative to the submitted v2 file (public LB 0.955), so the score
difference tells us where the validation->test gap comes from.

| File | Change | If score goes UP | If score goes DOWN |
|---|---|---|---|
| probeA_add_near_duplicates | +486k pairs: records whose closest S1 has the same name but a shifted house number / different legal form | test copies are more perturbed than train -> recall problem | they are sibling distractors -> look elsewhere |
| probeB_thr0.95 | -331k pairs (stricter threshold) | test has a precision problem | precision is fine |
| probeC_thr0.30 | +325k pairs (looser threshold) | recall problem on model-uncertain pairs | - |

## Results so far (public LB)
| File | Score |
|---|---|
| v2 baseline | 0.955 |
| probeA_add_near_duplicates | 0.894 (near-duplicates are distractors) |
| probeB_thr0.95 | 0.959 (precision problem confirmed) |

## Next: `matching_results_F1_sibling_veto.tsv.gz`
Baseline v2 + rule F1: reject an accepted pair when the record's house number is the S1's first
number + k (k in the offset set learned from train: 1,2,3,4,5,7,9,11,13,21), the S1's own number
is absent, AND (legal form differs OR a learned qualifier word is added OR k > 2).
Validation (train half B): 0.97824 -> 0.97875. Removes 0.124 / 0.039 / 0.082 accepted pairs per S1
(US / India / France) on test. Expected public LB about 0.965-0.969.
See research_forensics.md and research_france.md for the analysis.

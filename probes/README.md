# Leaderboard probe submissions (diagnostic only)

Each file is a full, validator-passing `matching_results.tsv` (gzipped). `gunzip -k <file>` before uploading.
They change exactly one thing relative to the submitted v2 file (public LB 0.955), so the score
difference tells us where the validation->test gap comes from.

| File | Change | If score goes UP | If score goes DOWN |
|---|---|---|---|
| probeA_add_near_duplicates | +486k pairs: records whose closest S1 has the same name but a shifted house number / different legal form | test copies are more perturbed than train -> recall problem | they are sibling distractors -> look elsewhere |
| probeB_thr0.95 | -331k pairs (stricter threshold) | test has a precision problem | precision is fine |
| probeC_thr0.30 | +325k pairs (looser threshold) | recall problem on model-uncertain pairs | - |

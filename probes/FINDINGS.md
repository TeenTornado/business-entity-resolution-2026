# Findings log (facts only, with the evidence behind each)

## Scores
| File | Validation half B macro F0.5 | Public |
|---|---|---|
| v4 | 0.9843 | 0.9757637 |
| v5 | 0.9843 | 0.9757612 |
| v6 (stage 1 + test self-training) | 0.98436 | 0.976378 |
| v6, France thr 0.4 | – | 0.975736 |
| v4, France empty (probe) | – | 0.84222 → France is about 0.951, others about 0.980 |
| **c1 = stage-2 collective (current package)** | **0.98769** | **0.978843** |
| c2 = c1 + a second collective pass | 0.98790 (half A unchanged) | not uploaded |
| c3 = c1 without 19 pool features | 0.98650 | not uploaded |

- Public = a subset of test. **Final ranking uses the private remainder** (README).
- The metric in `scoring.macro_f05` equals the official one: per S1, F = 1.25·TP/(0.25·T+P);
  T=0 and P=0 scores 1; macro average.

## What c1 is
- Stage-1 features: all 93 of `model_v6/decision.json` (**nothing removed**).
- Stage-1 is retrained as 2 out-of-fold models on orphan-free pairs (pairs whose record's
  true S1 lies outside the 800k+150k S1 subset are dropped: 810,839 pairs).
- Stage 2 adds 29 collective features (anchors = the S1's confident copies; the
  competitor's anchors) → `c_full/`. Threshold 0.7375, then the one-owner rule.
- Test features come from `test_pairfeat_v5` (the same candidate set as v6).
  `candidate_pairs.tsv` = 11,266,451 pairs (6.50/S1).
- c3 dropped: fq_*, vc_*, rn_*, nm_idf_extra_b, nm_max_idf_shared, ad_max_idf_shared,
  sb_grp_offk. c4 (proposed, not run) also drops rev_n, blk_n1 and sb_grp_b.

## Error structure (validation, c1)
- Pair precision 0.9981, pair recall 0.9672. Loss: misses 0.0105, false matches 0.0019.
- Empty-address true copies: 4.4 % of true copies; 97.7 % of empty-address records are
  true copies; 46 % of them belong to a name shared by 2+ S1 in the same country.
- 1.64 % of true pairs are never retrieved; the pruner loses about 0.1 % more.

## Distribution shift
- An adversarial classifier (val vs test pairs, US+India) reaches AUC 0.993 on all
  features, and 0.716 without the 19 pool features. The top shifted features are
  fq_addr_b_s23 (the empty-address count follows pool size: 222k train vs 112k test)
  and rn_name_eq_count (90th percentile: 6 on val, 29 on test).
- France parses no state at all (state_eq = -1 on every pair). Forcing state_eq=1 moves
  acceptance by only +1.3 %.
- France has 0.61 uncertain pairs (p in .1–.9) per S1; US/India have 0.25–0.27.

## Word-swap (WS) pairs
Definition: same first house number, address token-set ratio ≥ 90, core-name sets
differing by exactly 1 token on each side (not a typo: ratio < 60).

| | WS pairs / S1 | share that are true copies | c1 accepts |
|---|---|---|---|
| val US | 0.188 | 92.7 % | |
| val India | 0.151 | 75.4 % | |
| test US | 0.201 | ? | 83.3 % |
| test India | 0.190 | ? | 57.5 % |
| test France | 0.198 | ? | **36.7 %** |

- The claim that "test has far more WS distractors" is **not supported**: prevalence is
  about the same everywhere.
- In labelled data **WS pairs are mostly true copies** (word-substitution noise).
- France's WS acceptance is far below the US/India level. (First read: missed French true
  copies. **Superseded below:** inspection shows most rejected French pairs are real distractors.)

## Swapped-in word decides WS pairs (validation labels)
- True-copy noise uses a closed lexicon: center (7613 pairs, 99 % true), services (98 %),
  service (99 %), partners (99 %), mr, shri. Any other swapped-in word (solutions,
  technologies, industries, …) is about 0 % true → genuine co-located distractors.
- France test: the frequent swapped-in words are services (mean p 0.99), groupe (0.60),
  developpement (0.48), cie (0.82), associes (0.90), fils (0.45); each about 3.1–3.7k pairs,
  a flat block like a closed noise lexicon. Estimated upside if all are true copies:
  about +0.0003 overall.

## Pair-type table (val true-copy rate vs test acceptance)
| type | val true rate | US acc | France acc | France pairs/S1 |
|---|---|---|---|---|
| partial addr, same name | 0.89 | 0.97 | 0.77 | 0.85 |
| same addr, other name diff | 0.75 | 0.82 | 0.49 | 0.46 |
| same addr, 1-word swap | 0.92 | 0.90 | 0.54 | 0.28 |
| different house no., same name | 0.56 | 0.50 | 0.12 | 0.67 |
- US acceptance tracks the val true rates (calibrated). **Manual inspection of rejected
  France pairs shows mostly real distractors**:
  - same generic name ("Bordeaux Club SARL") on a different street;
  - siblings: house number offset by 4/9/13 with the qualifier **"France"** added (France's
    sibling qualifier);
  - invented brand names at the same address (ambiguous).
- ⇒ There is no evidence of a large France recall failure; the table cannot be read as
  missed copies, because France's distractor mix differs.

## Renamed records at the same address (no shared name token), val half B
- 0.247 pairs/S1, 75.5 % true copies; stage-1 recall on them 97.6 %, 57 FP.
  **Not a loss source.**

## Conclusions for experiment choice
- The "test has more same-address word-swap distractors" hypothesis is **rejected**.
- c1 keeps all features; "restoring features" (option C) = c1 itself.
- Self-training on test pseudo-labels is the one lever with public evidence (+0.0006 on
  v6). c1 does not use it yet → experiment E1.

## Web research (public repos for this challenge, 2026-09-26)
- No public write-up reports ≥ 0.98. Public scores found: Akash-bardia 0.9761;
  mayankgoplani431-del 0.964 (their best).
- mayankgoplani431-del lessons (validation → leaderboard):
  - A name-as-one-token TF-IDF pass + exact keys (name+house no., unique names,
    sorted-name+no.) raised blocking recall from 95.4 to 97.5 % → +0.011 public.
  - Density-sensitive group features: validation +0.003, **public −0.008**. They accepted
    about 230k extra test pairs, 60–90 % with a different house number (siblings). Test has
    5.8 pool records/S1 vs 4.67 in train. Same lesson as our c3 check.
- gojosatorou999: 4 retrieval channels with reciprocal-rank fusion; an expected-F0.5 prefix
  decision per entity vs the empty set; leave-one-country-out validation as a proxy for France.
- A community idea: a LoRA-fine-tuned ≤8B LLM (Qwen2.5-7B, Apache-2.0) as a re-ranker on
  borderline pairs. Needs a GPU; not feasible on our budget.

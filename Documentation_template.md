# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Local Aura Farmers
**Team Members:** Sreeram Kumar V R (Team Leader), Krishna Mohan S, Sricharan N S
**Submission Date:** 2026-09-26

---

## 1. Executive Summary

The pipeline has five stages:

1. **Normalisation:** country-agnostic, with a script-to-Latin dictionary learned from the
   training ground truth.
2. **Blocking:** recall-oriented, using sparse IDF cosine with conjunction keys plus an
   address-only channel. It keeps 98.4 % of true pairs.
3. **Learned candidate pruner:** cuts the candidate set to **6.5 candidates per S1 entity**
   while keeping 98.25 % of true pairs. This set is `candidate_pairs.tsv`.
4. **Stage 1:** a pairwise LightGBM matcher, trained on a training pool enriched with
   **simulated sibling distractors** that mimic the test set.
5. **Stage 2, a collective matcher:** it re-scores each pair by comparing the candidate
   record with the S1 entity's other confident copies and with the copies of its strongest
   competing S1 entity.

A tuned F0.5 threshold and a one-owner rule give the final lists.

| | Validation macro F0.5 | Public leaderboard |
|---|---|---|
| Stage 1 only | 0.9844 | 0.9764 |
| **Final (stage 1 + stage 2)** | **0.9877** | **0.9788** |

---

## 2. Methodology

### 2.1 Problem Analysis

| Observation (training data) | Number | Consequence |
|---|---|---|
| S1 entities with no match (singletons) | 5.6 % | An empty list must be predicted confidently; each singleton is worth a full 1.0 |
| Matches per S1 entity | mean 3.46, max 11 | Recall per entity dominates: one missed copy out of 3 costs about 0.09 F0.5 |
| S2/S3 records that belong to no S1 entity (distractors) | 26 % | About half are *siblings* (below); precision is critical |
| S2/S3 records that belong to more than one S1 | 0 | Enables the one-owner decision rule |
| True pairs whose country label differs | 0 % | Blocking runs within a country label (open set, so France works) |
| India S2/S3 names in an Indic script | 23.5 % | Needs transliteration; anyascii alone gives "marketimg", "praivet" |
| True copies with an **empty address** | 4.4 % | 97.7 % of empty-address records are true copies, but 46 % of them carry a name that 2 or more S1 entities in the country share |

**Sibling distractors.** About half of the unmatched records are near-copies of a real
entity:
- the house number is shifted by a small offset from a fixed set,
  K = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}, learned from train;
- the legal form is usually swapped;
- a qualifier ("Center", "Services", "Partners", …) is often added.

In the test pool they are about twice as frequent and appear in groups of 2–3 records.
The test pool also contains same-address records whose name differs by one substituted
word ("JG Federation" → "JG College").

Name noise:
- legal-form variation
- OCR/leet digits
- junk prefixes
- DBA and website forms
- bracketed or duplicated tokens
- dotted acronyms
- typos, and word-order changes

Address noise:
- abbreviations in English, French and Indian conventions
- component reordering
- "No./H.No./Door No./#/N°" prefixes
- zero-padding
- city aliases
- missing components and "null" fillers

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned pruner + two-stage gradient-boosted matcher + a global
assignment rule.

**Core Innovation:**
1. **Collective (group-aware) matching (stage 2).**
   - Every S1 entity has about 3.5 copies. Copies of one entity inside one source share
     that source's rendering of it: casing, spelling noise, address format.
   - A hard copy (for example one with an empty address, or a renamed one) often looks far
     more like the entity's *other* copies than like the S1 reference.
   - Stage 2 therefore compares each candidate with the S1 entity's confident copies
     ("anchors"). It also compares it with the anchors of the candidate's strongest
     competing S1 entity, which gives contrastive evidence about *which* same-name entity
     owns the record.
2. **Complete-universe training.**
   - The training pairs cover only a subset of S1 entities. In that subset, a record
     whose true owner falls outside the subset looks like a hard negative, which never
     happens on test, where every S1 entity is present.
   - Removing those "orphan" pairs made the training data behave like test. Stage-1
     recall on empty-address copies rose from 0.56 (earlier model, original validation) to 0.63, and to 0.70 after stage 2.
3. **Test-like distractor simulation.** 1.33 M sibling records were generated from training
   data only, using the learned offset set, legal-form mix and qualifier lexicon. They were
   added to the training pool.
4. **Learned transliteration dictionary** (1,347 word mappings, native-script state names)
   and **conjunction blocking keys** (`joined_name|address_token`, `name_token|house_number`).

---

## 3. Candidate Generation (Blocking)

For every record we build a bag of hashed tokens (2^24 buckets):

- `n:` name-core words (legal forms removed), adjacent word *joins*, and the full joined name
- `s:` the sorted joined name (tolerates word-order changes)
- `a:` normalised address words and numbers; `b:` address word bigrams within a comma component
- `j:` joined name × each address token; `x:` each name word × each house number

Within each **country label** (an open set, so France is handled like any other label):

1. Cap tokens that occur in more than 3,000 S2/S3 records.
2. Weight by IDF and L2-normalise.
3. Compute chunked sparse S1 × S2/S3ᵀ cosine scores with a numba top-k selection.
4. Keep the **top 30** per S1 entity, plus the **top 10 from an address-only channel**.
   The address channel recovers renamed and DBA copies.

A **learned pruner** then produces `candidate_pairs.tsv`. It is a LightGBM model on
cheap features: blocking scores and ranks, reverse competition, pool frequencies,
name/address cosines, vocabulary skew and sibling indicators. Pairs with q ≥ 0.004 are kept.

- **Blocking keys used:** name words/joins, sorted name, address words/bigrams, name×address and name×house-number conjunctions (IDF-weighted cosine), plus an address-only channel.
- **Candidate pairs generated:**
  - before pruning: train 74.3 M (33.7 per S1), test 58.5 M
  - **final candidate set (test): 11,266,451 pairs, 6.50 per S1 entity** (US 5.63, India 6.58, France 8.49)
- **Reduction ratio:** 1 − 11.27 M / 6.72 × 10¹² within-country comparisons = **99.99983 %**
- **Pair recall** (validation):
  - 98.36 % of true pairs after retrieval
  - **98.25 % after pruning**, at 5.7 candidates per S1
- **How we kept true matches from being lost:**
  - error analysis on the misses led to the conjunction keys and the address-only channel
  - frequent tokens are capped rather than removed
  - both the joined and the sorted name forms are indexed
  - transliteration is applied before blocking
  - the pruner threshold was chosen so it costs only 0.1 % of true pairs
  - Remaining misses are mostly name-only records with typos, and copies whose house number
    was changed.

---

## 4. Matching Model

**Stage-1 features (93):**
- **Name (~25):**
  - rapidfuzz ratio, token-sort, token-set and partial ratio on the legal-form-free core
  - Jaro-Winkler on the joined core
  - Jaccard and IDF-Jaccard, IDF coverage, fuzzy token coverage
  - first-token equality, containment, DBA alternatives
  - legal-form conflict, legal forms added or dropped, lengths, script flag
  - highest shared IDF, IDF mass of extra tokens
- **Address (~25):**
  - empty flags; token-set, token-sort and partial ratio; Jaccard and IDF-Jaccard; coverage
  - house-number counts, shared numbers, number Jaccard, first-number equality, conflict
  - house-number Levenshtein, prefix and log difference, in-range test
  - alpha-token overlap, state equality, component overlap
- **Candidate-graph context (12):** blocking score, rank, relative score, top gap; reverse count, rank, best competitor and margin; source flag
- **Pool frequency (8), vocabulary skew (4), name graph (3):**
  - how common the exact name/address is in each source
  - whether the extra name tokens are distractor-like qualifiers
  - how many of this S1's candidates share the exact name
- **Sibling features (~8):**
  - house-number offset in the learned set K, minimum offset, legal-form difference
  - whether a same-name group has members at different offsets

**Stage-2 (collective) features (29):**
- For the S1 entity's anchors (confident copies other than the candidate, p₁ ≥ 0.5 and
  argmax owner):
  - number of anchors, and number in the same source
  - best name token-set ratio, core-name ratio and exact core-name match
  - best **case-sensitive raw-name** ratio (overall and same source)
  - best address ratio and house-number Jaccard
- The same aggregates for the anchors of the candidate's **best competing S1 entity**.
- Differences between the two (contrastive), plus:
  - the stage-1 probability, its rank within the S1 entity
  - the competitor's probability and the margin over it
  - the number of S1 entities that retained the candidate
  - the empty-address flag, and whether the candidate is itself an anchor

**Model type:** LightGBM binary classifiers (MIT licence): pruner, stage-1 fold models and
the stage-2 model. Each has up to a few thousand trees × 127 leaves, far below the 8 B
parameter limit. No pretrained language model is used.

- **Training data:** candidates of 800,000 training S1 entities (about 5 M pairs after
  pruning).
- **Honest stage-1 scores:** stage-1 scores used to train stage 2 are *out-of-fold*. Two
  fold models are each trained on half of the training S1 entities and score the other
  half. Validation and test use the mean of the two.
- **Early stopping:** on validation half A.

**Threshold selection method:**
- Accept a pair if its stage-2 probability p ≥ t.
- Then apply the **one-owner rule**: an S2/S3 record accepted by several S1 entities stays
  only with the highest-probability one.
- t is found by grid search maximising macro F0.5 over all half-A S1 entities, singletons
  included: **t = 0.7375**.

Tried and not adopted:
- **Explicit sibling veto:** −0.0004 on validation.
- **A second collective pass:** +0.0002 on half B, no change on half A.
- **Dropping pool-statistic features:**
  - Validation vs test adversarial AUC fell from 0.993 to 0.716, but validation dropped
    0.0012.
  - Inspection showed the extra test matches were same-address, word-substituted
    distractors, so these features carry real signal.
- **A looser France threshold:** −0.0006 public.

---

## 5. Results & Error Analysis

Validation protocol:
- Training S1 entities are split at random: 800 k for fitting, then 75 k held out and
  split into half A (early stopping, threshold) and half B (reported).
- Candidates come from the full training pool, simulated siblings included.

| Version | Validation macro F0.5 | Public |
|---|---|---|
| v2: pairwise + graph-context + frequency features | 0.9782 | 0.955 |
| v4: + address channel, learned pruner, sibling features | 0.9843 | 0.9758 |
| v6: + sibling simulation, name graph, test self-training | 0.9844 | 0.9764 |
| **Final: + stage-2 collective matcher, orphan-free training** | **0.9877** | **0.9788** |

- **F_0.5 Score (macro):** **0.9877** on held-out training entities (half B); **0.9788** on the public leaderboard.
- Final validation breakdown:
  - pair precision 0.9981, pair recall 0.9672
  - loss from missed matches 0.0105; loss from false matches 0.0019
- **Stage 2 compared with stage 1 on the same folds:**

  | | Stage 1 | Stage 2 |
  |---|---|---|
  | Recall on empty-address copies | 0.633 | 0.699 |
  | Recall on other copies | 0.9913 | 0.9955 |
  | False pairs | 782 | 471 |

- **Common false positives:**
  - sibling groups (house number shifted by an offset in K, legal form swapped, qualifier added)
  - same-address records with a substituted name word
  - name-only records whose name is shared by several entities
- **Common false negatives:**
  - name-only (empty-address) copies of an entity whose name is shared by other S1 entities
  - true copies whose house number was changed
  - heavily rewritten names (DBA-only, e.g. "Novivio" for "Services Bright Lines Ltd")
  - records never retrieved (1.6 % of true pairs)
- **France (unseen in training):**
  - France has about 2.3× more uncertain pairs per S1 entity than US/India, mostly sibling
    groups and word-substituted distractors.
  - The collective features compare copies with each other rather than with dictionaries
    learned from US/India data, so they transfer without French training labels.

---

## 6. Conclusion

The largest single gain came from treating matching as a *group* problem. An S1 entity's
copies resemble one another, so a hard copy is best judged against the copies already
matched with confidence, and against the copies of its rival entities. Two other steps
mattered: training on a universe where every record's owner is present, and simulating the
test set's distractor patterns in training. Together they took the pipeline from 0.9764 to
0.9788 on the public leaderboard, with a validation score of 0.9877. It uses no external
data or large model. It handles the unseen France records through language-agnostic
normalisation and group features.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/`: all source code
- `model/`: shipped trained models
- `README.md`: exact reproduction steps and timings
- `requirements.txt`: pinned versions
- `run_pipeline.sh`: one-command end-to-end run

```bash
bash code/business_entity_resolution/run_pipeline.sh <student_resource/dataset> <work_dir> output
```

| Module | Role |
|---|---|
| `normalize.py` | legal forms, EN/FR/IN address abbreviations, ordinals, OCR digits, DBA/website handling, state tags, transliteration hook |
| `learn_translit.py` | script→Latin dictionary from train pairs |
| `preprocess.py` | parallel parsing of all records |
| `simulate_siblings.py` | learns offsets/qualifiers/legal mix and simulates sibling distractor groups in the train pool |
| `blocking.py` | hashed IDF cosine blocking + address channel, numba top-k |
| `features.py` | pairwise string features |
| `scoring.py` | graph-context, frequency, vocabulary, sibling and name-graph features; decision rule; macro F0.5 |
| `train.py` / `predict.py` | pruner + stage-1 training/tuning; per-country test candidate stage and scoring |
| `collective.py` | stage-2 collective matcher (out-of-fold stage 1, group features, final decision, writer) |

### B. Additional Results

Test-set predictions (final submission):

| Country | S1 entities | Candidate pairs | Candidates / S1 | Predicted matches | Matches / S1 |
|---|---|---|---|---|---|
| US | 663,106 | 3,735,756 | 5.63 | 2,230,710 | 3.36 |
| India | 809,986 | 5,328,740 | 6.58 | 2,697,142 | 3.33 |
| France (unseen in training) | 259,452 | 2,201,955 | 8.49 | 825,213 | 3.18 |
| **Total** | **1,732,544** | **11,266,451** | **6.50** | **5,753,065** | **3.32** |

- 94.0 % of test S1 entities receive at least one match (train: 94.4 % are non-singletons).
- Both files pass `utils/validate_submission.py`.

Top stage-2 features by gain:
1. stage-1 margin over the best competing S1 entity
2. stage-1 probability
3. anchor flag
4. address similarity difference (own anchors vs competitor's anchors)
5. case-sensitive raw-name similarity to same-source anchors
6. number of same-source anchors
7. reverse-competition margin
8. house-number Jaccard with anchors

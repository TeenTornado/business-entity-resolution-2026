# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Local Aura Farmers
**Team Members:** Sreeram Kumar V R (Team Leader), Krishna Mohan S, Sricharan N S
**Submission Date:** 2026-09-25

---

## 1. Executive Summary

The pipeline has three stages. The first normalises each record in a country-agnostic
way, using a script-to-Latin word dictionary learned from the training ground truth. The
second is a sparse IDF-cosine blocking step with *conjunction keys*. It keeps
**98.1 % of all true pairs** while scoring only 30 candidates per Source-1 entity, a
reduction ratio of 99.9994 %. The third is a LightGBM matcher over 61 features, with an
F0.5-tuned threshold and a *one-owner* rule (each S2/S3 record goes to at most one S1
entity). On 75,000 held-out training S1 entities that were never used for fitting or
early stopping, the pipeline reaches **macro F0.5 = 0.9782**.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on the training data:

| Observation | Number | Consequence |
|---|---|---|
| S1 entities with no match (singletons) | 5.6 % | "Empty" must be predicted confidently; each singleton is worth a full 1.0 |
| Matches per S1 entity | mean 3.46, max 11 | Recall per entity matters; one missed match costs about 0.06 F0.5 |
| S2/S3 records that belong to no S1 entity (distractors) | 2.68 M of 10.3 M (26 %) | Many near-duplicates of real entities; precision is critical |
| S2/S3 records that belong to **more than one** S1 | 0 | Enables the one-owner decision rule |
| True pairs whose country label differs | 0 % | Blocking can run within a country label (open set, so France works) |
| India S2/S3 names written in an Indic script (Devanagari, Tamil, Telugu, Kannada, Bengali, …) | 23.5 % | Needs transliteration. anyascii alone gives "marketimg", "praivet" |
| Addresses with a state name written in native script | 9 % of pairs | Mapped back with a learned component dictionary |
| S2/S3 records with an empty address | 3.3 % | Must match on name only, so name-frequency features are needed |

Name noise we saw:
- legal-form variation (Pvt/Private, Ltd/Limited, L.L.C., prefixed "Private X Limited")
- OCR/leet digits ("Chemica1s", "5ervices", "Leapfr0g")
- junk prefixes ("--", "<<", "***")
- DBA forms ("Synkor d/b/a Gugliotta …", "Arcsolx dba Kumar & Co")
- website/hashtag forms ("kempgloba.com", "#bellmodern")
- bracketed or duplicated tokens
- typos, and word-order changes

Address noise we saw:
- Rd/Road, Ave/Avenue, Bd/Boulevard, R/Rue, "Ninth"/"9th"
- component reordering
- "No./H.No./Door No./#/N°" prefixes
- zero-padding ("AF-0684")
- city aliases (Bombay/Mumbai, Bangalore/Bengaluru)
- missing components, and "null"/"N/A" fillers

### 2.2 Solution Strategy

**Approach Type:** Blocking + gradient-boosted pairwise classifier + a global assignment rule.

**Core Innovation:**
1. A **transliteration dictionary learned from training pairs.** When an Indic-script
   S2/S3 name has the same number of words as its matched Latin S1 name, we align the words
   by position and take a majority vote. This gives 1,347 word mappings (e.g.
   मार्केटिंग → marketing, प्रा. लि. → pvt ltd) and 16 native-script state names. Unseen
   words fall back to anyascii.
2. **Conjunction blocking keys:** `joined_name|address_token` and `name_token|house_number`.
   They stay selective even when every individual word is common, which is typical of
   Indian names such as "Vision Solutions Private Limited".
3. **Candidate-graph features** (reverse rank and score margin against competing S1
   entities for the same S2/S3 record) plus **pool-frequency features**. They are the
   strongest predictors and make the matcher aware of distractors.

---

## 3. Candidate Generation (Blocking)

For every record we build a bag of hashed tokens (2^24 buckets):

- `n:` name-core words (legal forms removed), adjacent word *joins* ("kemp globa" →
  "kempgloba", which matches the website form), and the full joined name
- `s:` the sorted joined name (tolerates word-order changes)
- `a:` normalised address words and numbers; `b:` address word bigrams within a comma component
- `j:` joined name × each address token; `x:` each name word × each house number (conjunction keys)

Within each **country label** (an open set: the loop runs over whatever labels appear in
S1, so France is handled like any other label), we do the following:

1. Drop tokens that occur in more than **3,000** S2/S3 records. These are too common to
   be selective and would blow up the sparse product.
2. Weight the remaining tokens by IDF and L2-normalise each record's vector.
3. Compute S1 × S2/S3ᵀ cosine scores with chunked `scipy.sparse` products across 4
   processes.
4. Keep the **top 30** per S1 entity, using a numba top-k selection.

- **Blocking keys used:** name words/joins, sorted name, address words/bigrams, name×address and name×house-number conjunctions (IDF-weighted cosine).
- **Candidate pairs generated:**
  - train: 66.2 M pairs for 2.21 M S1 entities
  - test: **51.98 M pairs** for 1.73 M S1 entities (US 19.9 M, India 24.3 M, France 7.8 M)
- **Reduction ratio:** 1 − 66.2 M / 11.8 × 10¹² within-country comparisons = **99.99944 %**
- **Pair recall** (fraction of all GT pairs present in candidates, full train set): **98.14 %**.
  The recall@K curve measured while designing the blocking (US / India):
  - @10: 95.8 / 96.2 %
  - @30: 98.1 / 98.2 %
  - @100: 98.9 / 98.9 %
- **How we kept true matches from being lost:**
  - We did error analysis on the misses and added the conjunction keys, which raised
    India recall@30 from 96.7 % to 97.9 %.
  - Frequent tokens are capped instead of removed entirely.
  - Both the joined and the sorted name forms are indexed.
  - Transliteration is applied *before* blocking.
  - The remaining misses are mostly records with an empty address whose name is also
    common, or whose name is completely different (DBA-only). A precision-oriented
    matcher would reject most of them anyway.

`output/candidate_pairs.tsv` holds exactly these 30-per-entity lists. They are the exact
input to the model, and every predicted match is a subset of them.

---

## 4. Matching Model

**Features used (61):**
- **Name (23):**
  - rapidfuzz ratio, token-sort, token-set and partial ratio on the legal-form-free core
  - Jaro-Winkler and ratio on the space-free joined core; ratio on the full name
  - Jaccard and IDF-weighted Jaccard, IDF coverage of the S1 name
  - fuzzy token coverage in both directions (typo-tolerant)
  - first-token equality, containment, best DBA-alternative similarity
  - legal-form conflict and shared legal forms; lengths
  - script-origin flag; highest IDF among shared tokens; IDF mass of extra tokens in the S2/S3 name
- **Address (19):**
  - empty flag; token-set, token-sort and partial ratio; Jaccard and IDF-Jaccard
  - IDF coverage in both directions; fuzzy coverage
  - house-number counts, shared numbers, number Jaccard, first-number equality, **number conflict**
  - state equality (full names and abbreviations unified, native script mapped)
  - comma-component overlap; highest shared IDF
  - one combined name+address token-set ratio
- **Candidate-graph context (10):**
  - blocking score, rank, score relative to the S1 entity's best, gap from #1 to #2, number of candidates
  - **reverse** features: how many S1 entities chose this S2/S3 record, this S1's rank among
    them, the best competing S1 score, and the **margin** over it
  - source flag (S2/S3)
- **Pool-frequency (8):**
  - how many S1 / S2/S3 records in the same country share the exact core name (for each side) or the exact address
  - how many of this S1's candidates share the candidate's exact name

  These let the model trust a name-only match when the name is unique, and distrust it when the name is generic.

**Model type:** LightGBM binary classifier (MIT licence, about 940 trees × 127 leaves; far
below the 8 B-parameter limit; no pretrained language model).
- Training data: all top-30 candidates of 300,000 training S1 entities (9.0 M pairs, 0.34 M positives).
- Early stopping on a separate half of the held-out entities.

**Threshold selection method:**
- Rule: accept a pair if p ≥ t, then apply the **one-owner rule**. If an S2/S3 record is
  accepted by several S1 entities, it stays only with the highest-probability one.
- t is chosen by grid search maximising **macro F0.5 over all held-out S1 entities**,
  singletons included.
- The optimum is flat between t = 0.65 and 0.80 (F0.5 ≥ 0.9780). Chosen **t = 0.713**.

Tried and not adopted:
- **Per-entity expected-F0.5 subset selection** (Monte-Carlo over the probabilities):
  0.9764 vs 0.9770, worse than a global threshold.
- **Stage-2 re-scoring on competing probabilities:** +0.00015 on validation. Not
  adopted, because in validation its competitor features only covered a subset of S1
  entities, so the gain might not carry over to the full test graph.

---

## 5. Results & Error Analysis

Validation protocol:
- The 2.21 M training S1 entities are split at random into 300 k for training, 75 k
  (half A) for early stopping, and 75 k (half B) for the threshold and the reported score.
- Candidates always come from the **full** training S2/S3 pool, so distractor density is realistic.

| Version | Validation macro F0.5 |
|---|---|
| v1: pairwise + graph-context features | 0.9770 |
| v1 without the one-owner rule, p ≥ 0.5 | 0.9749 |
| **v2: + pool-frequency features (final)** | **0.9782** |

- **F_0.5 Score (macro):** **0.9782** on held-out training entities (US 0.977, India 0.977 in v1; the two countries are balanced).
- **Where the remaining loss comes from (v1 analysis):**
  - about 80 % comes from missed matches
    - 1/3 of those are outside the candidate set
    - 2/3 are in the candidates but below the threshold
  - about 20 % comes from false merges
  - 2.2 % of singletons receive a false match
- **Common false positives (wrong merges):**
  - distractor records that are near-exact copies of the S1 entity but have a slightly
    different house number (237 vs 242 Herlong Ave) or a different city in the same state
    (Chennai vs Tiruchirapalli)
  - name-only S2/S3 records ("B Y ATHENA LLC CENTER", empty address) whose name overlaps a real entity
  - some look like label noise: identical name and address but labelled as a non-match
- **Common false negatives (missed matches):**
  - S2/S3 records with an **empty address** and a generic name ("Continental Society",
    "Valley Network Associates")
  - heavy name rewrites ("Management Hind Merchandising Limited Services")
  - partial addresses that only share a city
  - completely different trade names (DBA-only records such as "Lyradelta") with a
    changed house number

**Test predictions:** see Appendix B.

---

## 6. Conclusion

In this data, careful normalisation and recall-oriented blocking with conjunction keys
matter as much as the classifier. The strongest single signals came from the candidate
graph (does another S1 entity claim this record more strongly?) and from name/address
frequency in the pool. Together with the one-owner rule and a tuned F0.5 threshold, a
compact LightGBM model reaches 0.978 macro F0.5 on held-out data. It uses no external
data and no large model, and it handles the unseen France records through
language-agnostic normalisation and features.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/`: all source code
- `README.md`: exact reproduction steps and timings
- `requirements.txt`: pinned versions
- `run_pipeline.sh`: one-command end-to-end run

Entry point:

```bash
bash code/business_entity_resolution/run_pipeline.sh <student_resource/dataset> <work_dir> output
```

This regenerates `output/candidate_pairs.tsv` and `output/matching_results.tsv` and runs
the official validator.

| Module | Role |
|---|---|
| `normalize.py` | legal forms, address abbreviations (EN/FR/IN), ordinals, OCR digits, DBA/website handling, state tags, transliteration hook |
| `learn_translit.py` | script→Latin dictionary from train pairs |
| `preprocess.py` | parallel parsing of all records |
| `blocking.py` | hashed IDF cosine blocking, numba top-k |
| `features.py` | 43 pairwise string features |
| `scoring.py` | graph-context and frequency features, one-owner decision rule, macro F0.5 |
| `train.py` / `predict.py` | training/tuning and block-wise test inference |

### B. Additional Results

Threshold sweep, stage 1, validation half B (macro F0.5):

| t | 0.30 | 0.40 | 0.50 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| F0.5 | 0.9688 | 0.9734 | 0.9761 | 0.9776 | 0.9780 | 0.9782 | 0.9781 | 0.9778 | 0.9756 | 0.9716 |

Top features by gain:
1. reverse score margin
2. reverse rank
3. address token-set ratio
4. IDF mass of extra name tokens
5. house-number conflict
6. full-name ratio
7. name+address token-set
8. number Jaccard
9. legal-form conflict
10. S2/S3 address frequency

Test-set predictions, as a sanity check. Train ground truth has 3.46 true matches per S1
entity and 94.4 % of entities are non-singletons, which predicts about 3.35 per entity
once candidate recall is accounted for.

| Country | S1 entities | Candidate pairs | Predicted matches | Matches / entity |
|---|---|---|---|---|
| US | 663,106 | 19.89 M | 2.27 M | 3.42 |
| India | 809,986 | 24.30 M | 2.70 M | 3.33 |
| France (unseen in training) | 259,452 | 7.78 M | 0.83 M | 3.19 |
| **Total** | **1,732,544** | **51.98 M** | **5.79 M** | **3.35** |

- 94.2 % of test S1 entities receive at least one match; 100,401 are predicted as singletons.
- Both files pass `utils/validate_submission.py`, including `--check-ids`.

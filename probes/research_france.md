# Selective sibling veto, and France specifics

**Data used**
- Validation: half-B (75,102 S1), with exact macro F0.5 from `scoring.macro_f05`. The baseline is 0.97824.
- Test: a random 35% of S1s (605,598 S1: US 231,877, India 282,880, France 90,841). The decision rule was computed on the full candidate set first, then sampled.
- Qualifier list: learned on 100k train-split S1s (1.5M candidate pairs, of which 335,750 are true).
- Scripts and the learned list are in `/home/user/work/research_france_scripts/`. `lib.py` holds the features, `rules.py` the rules, and `final_def.py` the exact rule below.

Notation:
- A: the digit runs of the S1 `a_nums`. B: the digit runs of the record's `a_nums`, extracted with regex `\d+`, keeping runs of at most 6 digits. This also fixes the glued "No123" case.
- K = {1,2,3,4,5,7,9,11,13,21}.
- **off1** ("offset-first"): A[0] ∉ B, and some x ∈ B \ A has x − A[0] ∈ K.

## 0. Where the test false merges are
Accepted pairs per S1, by number relation. Val (all / true) vs test:

| rel | US val | US test | India val | India test | France test |
|---|---|---|---|---|---|
| exact (A[0] ∈ B) | 2.682 / 2.679 | 2.659 | 2.724 / 2.715 | 2.705 | 2.814 |
| no number | 0.338 / 0.335 | 0.346 | 0.418 / 0.413 | 0.427 | 0.254 |
| other number | 0.265 / 0.263 | 0.271 | 0.141 / 0.139 | 0.140 | 0.034 |
| **offset (k ∈ K)** | 0.022 / **0.017** | **0.139** | 0.027 / **0.025** | **0.063** | **0.089** |

- All of the test excess is in the offset class. The other classes match validation within ±0.02.
- In validation, offset pairs are mostly **true** (83%). These are number typos: k=1,2 are 89-97% true, and offsets on non-first numbers in India are about 99% true.
- True offset copies also come in groups. In val, 97-100% of offset groups with ≥2 accepted records are true, and 84% of true offset pairs have an exact-number sibling copy as well. So the "S1 has an exact copy" and "separate number group" signals **do not separate** true from false in validation.

Cell analysis of offset-first pairs, per S1 (val true → test):
- US
  - legal change and k>2: 0.0002 → 0.0886
  - legal change and k≤2: 0.0023 → 0.0254
  - no legal change, no qualifier, k>2: 0.0018 → 0.0090
  - **no legal change, no qualifier, k≤2: 0.0097 → 0.0116. This cell is typos, so leave it.**
- India: 0.0004 → 0.0225, 0.0026 → 0.0083, 0.0014 → 0.0041, 0.0091 → 0.0100.

## A. Rules
**Fixed legal form (lgF).**
- L(x) = tokens(n_legal) − {and, the, of, dba, france, india, usa, us}.
- For France S1, the French forms are re-extracted from raw_name with `(?<![a-z0-9])(s\.a\.r\.l|s\.a\.s\.u|s\.a\.s|e\.u\.r\.l|s\.c\.i|s\.n\.c|e\.i|s\.a|sarl|sasu|sas|eurl|sci|snc|ei|sa)\.?(?![a-z0-9])` (lower-cased, dots dropped).
- This also fixes two new bugs found here. **'sci' is missing from LEGAL_CORE_DROP**, so it stays in the core name. **Dotted S.A.S.U./E.U.R.L./S.C.I./S.N.C./E.I. are not parsed**, so the record gains the core tokens u / e l r u / c n s.
- lgF_diff = L(S1) ≠ L(rec). This covers a swap, an add or a drop.

**Qualifier (qual).** added = (core ∪ legal tokens of the record) − (those of the S1) − pure legal forms. For France, single-letter tokens are also removed. qual = added ∩ Q ≠ ∅.
- **Q_train (learned on TRAIN):** tokens added in ≥15 sibling-like distractors (label 0, offset, name Jaccard ≥0.5; 77,671 pairs), with a true-add rate ≤2e-5 and a likelihood ratio ≥20. This gives 101 tokens: holdings group industries ventures infratech overseas exports north/south/east/west …side/…gate summit highland harbor riverside metro valley midtown uptown downtown central greater lakeside coastal solutions technologies foods engineering hotel trading … (full list in `Q_train.parquet`). They cover 34% of sibling distractors and 0.0066% of true pairs.
- **France additions (learned label-free on test):** tokens added far more often in offset pairs than in exact-number pairs (ratio ≥20): france, participations, distribution, international, holdings.
  - This proxy was checked on train: all 25 of the 25 proxy tokens are also in the labelled list.
  - **Groupe and "& Fils" are not sibling markers.** 88% and 94% of their additions are on exact-number pairs, and "Et Fils", Groupe, Associés, Cie and Services replace a name word the way Center/Services/Partners do in US true copies.

Scoring columns:
- ΔF val is exact.
- E1 is the requested estimate (test true-rate = val true-rate per country; France uses pooled US+India).
- E2 assumes that true pairs removed per S1 are the same as in val (count invariance), and is simulated per S1 with the full F0.5 formula. The linear 0.19/0.04 version gives the same result to within ±0.0005.

| Rule (on accepted pairs) | val n/S1 | val true | ΔF val | test n/S1 US / IN / FR | ΔF test E1 | **ΔF test E2** |
|---|---|---|---|---|---|---|
| **F1 off1 & (lgF_diff \| qual \| k>2)** | 0.0078 | 0.56 | **+0.0005** | 0.125 / 0.039 / 0.088 | +0.0021 | **+0.0136** |
| F1a off1 & (lgF_diff \| qual) | 0.0047 | 0.58 | +0.0004 | 0.116 / 0.035 / 0.082 | +0.0019 | +0.0129 |
| off1 & lgF_diff (legal change only) | 0.0047 | 0.58 | +0.0004 | 0.114 / 0.032 / 0.053 | +0.0018 | +0.0117 |
| off1 & qual (qualifier only) | 0.0000 | n/a | 0 | 0.002 / 0.004 / 0.032 | +0.0015 | +0.0015 |
| off1 & k>2 | 0.0049 | 0.40 | +0.0006 | 0.099 / 0.030 / 0.070 | +0.0043 | +0.0112 |
| F1 & S1 has accepted exact copy | 0.0066 | 0.56 | +0.0003 | 0.116 / 0.037 / 0.082 | +0.0033 | +0.0133 |
| F1a & offset group is a singleton | 0.0024 | 0.24 | +0.0005 | 0.075 / 0.028 / 0.070 | +0.0066 | +0.0102 |
| F1 + propagate to whole offset group | 0.0099 | 0.65 | +0.0003 | 0.128 / 0.041 / 0.088 | +0.0001 | +0.0133 |
| offset & separate group ≥2 & S1 has exact | 0.0115 | **0.99** | −0.0010 | 0.046 / 0.021 / 0.012 | −0.0021 | +0.0019 |
| naive veto (offset & S1 has exact) | 0.0203 | 0.83 | −0.0009 | 0.128 / 0.058 / 0.084 | −0.0023 | +0.0121 |
| all offset pairs | 0.0240 | 0.83 | −0.0015 | 0.139 / 0.063 / 0.089 | −0.0051 | +0.0114 |
| qualifier on non-offset pair | 0.0004 | 0.31 | +0.0001 | 0.001 / 0.007 / **0.063** | +0.0014 | +0.0027 |
| lgF swap on non-offset pair | 0.0096 | 0.95 | −0.0008 | 0.017 / 0.003 / 0.006 | −0.0007 | −0.0005 |

**F1 per country**
- Val: US 0.0090/S1 (48% true, +0.0009), India 0.0060/S1 (73% true, +0.0000). True pairs removed: 0.0044/S1 in each.
- Test, E2 false positives removed per S1: US 0.121, India 0.034, France 0.083.
- Test ΔF: US +0.022, India +0.006, France +0.016 under E2; US +0.006, India −0.001, France +0.002 under E1.
- Of the pairs F1 flags, 26-40% have p ≥ 0.95, so a threshold cannot reach them. 93% have an exact-number copy, and 65-85% are in singleton offset groups.

**E1 is too pessimistic.** Val has only 0.004 false positives per S1 in this class, so E1 implies 7-14× more true offset pairs in test than in val. That contradicts the class counts in §0.

**Check of the estimator against the leaderboard.** For the threshold 0.713→0.95 probe:
- A p-band estimate predicts +0.012. It overshoots because true copies have lower p in test: 0.050 vs 0.023 exact-number pairs per S1 fall in [0.713, 0.95).
- The class-level count-invariance estimate predicts +0.004 to +0.007. The observed gain was **+0.004**.
- **Realistic expectation for F1: +0.010 to +0.014, i.e. a leaderboard score of about 0.965-0.969.**

## B. France (accepted pairs, 3.19/S1)
| Record vs S1 | share of accepted | per S1 | of which offset | exact-number |
|---|---|---|---|---|
| adds "France" (word or bracket) | 1.73% | 0.055 | 53% | 43% |
| adds "(France)" bracketed only | 0.03% | 0.001 | 14% | 77% |
| adds Groupe | 0.26% | 0.008 | 5% | 88% |
| adds "& Fils"/"Et Fils" | 0.88% | 0.028 | 0.2% | 94% |
| adds International / Participations | 0.01% / 0.03% | 0.0003 / 0.001 | 47% / 53% | ~50% |
| any of those 5 qualifiers | 2.91% | 0.093 | **33%** | 62% |
| French legal form **swapped** (both have one, disjoint) | 0.76% | 0.024 | **78%** | 20% |
| French legal form added (S1 has none) | 6.91% | 0.221 | 15% | 77% |
| qualifier or swap | 3.66% | 0.117 | 42% (0.049/S1) | 54% |
| qualifier, swap or add | 10.5% | 0.335 | 24% (0.080/S1) | 69% |

- France offset accepted pairs are 0.089/S1 (2.8% of accepted). 90% of them carry a qualifier or a legal change, and F1 flags 98% of them (0.0875/S1).
- The legal bug hides French swaps. 21% of accepted France off1 pairs are true swaps, and the buggy `lg_conflict` fires on 1.5%. Across all accepted pairs, the bug masks 0.019 swaps per S1 (US 0.017, India 0.003).
- The legal-form bug is itself the selection effect: among accepted offset pairs, the buggy lg_conflict is about 0 in every country. The model rejects the swaps it can see, and the masked ones are what get through.
- Exact-number France pairs with an added qualifier (0.06/S1) look like France's version of the true-copy "word → Services/Center" noise. France non-offset acceptance (3.10/S1) is no higher than US/India (3.27-3.28). **Do not veto them.**

## C. Ranked recommendations
1. **Ship F1 as post-processing, applied after `decide()`.** Reject an accepted pair when:
   - A and B are non-empty, A[0] ∉ B, and ∃x ∈ B∖A with k = x − A[0] ∈ K,
   - **and** at least one of: lgF_diff; qual (Q_train ∪ {france, participations, distribution, international, holdings}); or min such k > 2.
   - Estimated effect: val +0.0005, test **+0.010 to +0.014** (E2 +0.0136). Do not add the exact-copy or group conditions: they are neutral or worse. Do not reassign the vetoed record. Keep thr=0.713; after F1 the 0.95 threshold would mostly remove true pairs.
   - Safer fallback: off1 & k>2. Val +0.0006, test E2 +0.011, and the best E1 (+0.004).
2. **Fix normalisation and retrain** (these should be model features, not rules):
   - Remove {and, the, of, dba, france, india, usa, us} from n_legal. Add `sci` to LEGAL_CORE_DROP. Parse the dotted French forms. Then recompute lg_conflict and lg_shared, and add lg_change_type (swap/add/drop).
   - Add num_rel (exact / offset-first / offset-other / other / none), the signed first-number delta, an in-K flag, and the interaction offset × lg_change.
   - Add a qualifier score: the maximum log-ratio of added tokens, using Q_train plus the France proxy list.
3. **Group features only with augmentation.** Offset-group size, "separate from the exact group" and has_exact have the **opposite sign in train** (99% true) and in test. They should only be added after injecting test-like sibling groups (1-3 copies, +k, legal swap about 0.85, qualifier about 0.4, including French qualifiers) into train and validation.
4. **Don't:**
   - the naive veto, or vetoing all offset pairs (val −0.001 to −0.0015);
   - qualifier or legal-swap vetoes on non-offset pairs (France 0.063/S1 at risk);
   - treating Groupe or "& Fils" as qualifiers.

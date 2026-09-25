# Data forensics: why TRAIN-val scores 0.978 but TEST scores 0.955

All statistics come from random samples: 60k S2/S3 records per split (train was stratified 30k matched and 30k unmatched), 20k S1 per country, and 30k records per cell for operator rates. No entity-id or row-order signal was used. The scripts are in `/home/user/work/forensics_scripts/`.
Notation: a **sibling** is a distractor that is a perturbed near-copy of a real entity. The **offset signature** means that one house-number digit run is the S1 value plus k, where k is one of {1,2,3,4,5,7,9,11,13,21}.

## TL;DR
1. The extra records in test are **distractors, not more copies.** The number of true copies per S1 is unchanged (about 3.42 in test vs 3.46 in train). Distractors per S1 nearly double, from 1.22 to about 2.3.
2. About half of all distractors are **siblings**: the same street and the same name, with the house number increased by k from {1,2,3,4,5,7,9,11,13,21}. The legal form is usually swapped too, and a qualifier word is often added.
3. The key structural change is that **test siblings come in clusters of 2-3 noisy records.** In train they are almost always single records. The model learned that a repeated address means a true cluster, so it accepts test sibling clusters. It also has never seen the French qualifier words.
4. Accepted sibling-signature pairs in test: US 0.129, France 0.085, India 0.076 per S1, against a total validation FP of 0.016 per S1. The estimated F0.5 loss is US 0.027, France 0.020, India 0.014, or **about 0.020 weighted. That explains most of the 0.023 gap.** Recall looks unchanged.

## 1. What the unmatched TRAIN records are (2.68M records, 26% of S2/S3)
- The two sources hold exactly equal numbers of unmatched records: S2 1,340,997 and S3 1,340,857, which is 0.6077 per S1 each, in both US and India. True copies are not balanced this way (S2 1.674 vs S3 1.788 per S1). So distractors are drawn per source at a fixed rate and are not orphan entities copied like true ones.
- **(a) Orphan clusters: essentially absent.** Only 2.4% of unmatched records have even one strong unmatched S2/S3 neighbour (name 3-gram Jaccard ≥0.8 and address ≥0.5), and never two or more. With the same key search, 52.6% of matched records have at least one same-entity neighbour. So orphan clusters are at most about 5% of distractors.
- **(b) Siblings: about 52% in US and 38-45% in India.**
  - 59.9% of unmatched records have an S1 candidate whose first number differs by k from the offset set. The chance rate for matched records is 7.3%. With a strict address condition the figures are 51.3% vs 1.8%.
  - The offset histogram is flat over {1,2,3,4,5,7,9,11,13,21} (about 500 each out of 5,900), is always positive, and other values are almost absent. This is a generator fingerprint.
  - Only 0.44% (US) and 1.0% (India) of true pairs carry the signature.
  - The legal form changes in 87% of siblings vs 28.5% of true copies.
  - Added qualifier words never appear as additions in true copies: Holdings 6.9%, Group 6.8%, Industries, Ventures, Infratech, Exports, Overseas, Enterprises, "Public", North/South/East/West, Northside/Southside/Eastgate/Westgate, Metro, Greater, Summit, Central, Uptown/Midtown/Downtown, Lakeside, Valley, Coastal, Harbor, Riverside, Highland. About 40% of siblings get one.
  - Some siblings also swap a name word (e.g. "Jones Activate" for "Torres Activate") or append Partners, Corp or Co.
- **(c) Unrelated singletons: the remaining 45-55%.** They have no close S1 and no cluster-mates. Their names use the same distractor template (qualifier words: US 44%, India 35%, vs 6-8% in true copies).
- Exact duplicates of an S1 record (same name, street and number): 1.3%.

## 2. Noise operators (rate %, train, matched records; unmatched in brackets)
| Operator | S2 | S3 |
|---|---|---|
| UPPERCASE address (state left in title case) | 90 [94] | 0.5 |
| US state | abbreviation | full name 89 [99.6] |
| India state | full name, 23% Indic script | abbreviation 65-73%, 22% Indic script |
| "Unit ..." kept | 0 | 5.8 (S1 13.4) |
| PO Box / PMB added (US) | 1.6 | 3.0 |
| "#" prefix (US / India) | 4.9 / 10.4 | 9.6 / 9.4 |
| NO. prefix (India) | 6.7 [12.3] | 6.6 [11.7] |
| null, N/A component | 2.9 | 2.7 |
| Number range, leading zero, trailing "." or "-" | 3 / 4.5 / 3.6 | same |
| City suffix CDP / CITY / Township | 9.7 | 9.3 |
| Empty address | **5.2 [0.3]** | **4.5 [0.3]** |
| UPPERCASE name | 23.6 [16.5] | 3.1 |
| lowercase name | 7.7 [3.2] | 7.8 [3.8] |
| Indic-script name (India) | 23 | 13 |
| Honorific Mr/Dr/Smt/M/s (India only) | 7.8 | 8.0 |
| Website / @handle | **5.8 [0.6]** | **5.3 [0.7]** |
| DBA / "formerly" / "née" | 0 | **3.6 [0.0]** (S3 only) |
| Legal form first ("LLC Moncada ...") | 2.0 [4.1] | 2.1 [4.1] |
| Bracketed legal form, double space, hyphen join | 7 / 12 / 5 | same |
| Duplicated token, digit substitution (6illy, L0gistics), junk prefix (-- >> ***), "(ID: n)" | 2.3 / 3.0 / 1.1 / 0.4 | same |
| Name replaced by a pseudo-word or scrambled | about 3.4% of true pairs (name 3-gram Jaccard <0.25) | |

Other operators seen: typos, token drop or shuffle, accents (Ínc, Có), and a "Center"/"Services" suffix (6% [4.6]).

**Asymmetries:**
- Only or mostly on true matches: DBA / formerly / née, website or handle forms (10x), empty address (17x).
- Only or mostly on distractors: the offset signature, legal-form change combined with a number change, qualifier words (7x), legal form first (2x).
- Noise in general is lighter on distractors: lowercase, hyphen and digit-substitution rates are 0.5-0.7x.

**France in test:**
- Several operators are absent: junk prefix, "##", null component, "(ID:)", number ranges, duplicated tokens. Digit substitution is 0.4%.
- It adds "N°", "R."/"Bd"/"Av." abbreviations, and département names in place of région names.
- The English qualifier words appear 0% of the time. French qualifiers appear instead ("(France)", France, Groupe, International, "& Fils"/"Et Fils", Participations): 19.8% of S2/S3 names vs 10.7% of S1 names.

## 3. Test composition
- **Count invariant.** Assume the train match rates per S1 and equal unmatched counts per source. The implied distractors per source per S1 are then, from S2 and from S3: US 1.148 / 1.147, India 1.181 / 1.182, France 1.037 / 1.032. The two estimates agree to within 0.005. The fitted copy multiplier is a = 0.99 (US), 1.00 (India), 0.96 (France).
- **Label-free mixture fits (US and India).** Estimates of the unmatched share from the copy-link rate, qualifier rate, sibling signature and website/empty-address rates all come out at 0.39-0.43. The prediction is 0.40 if the extra mass is distractors, and 0.21 if it is extra copies.
- **Sibling-signature records per S1** (candidates with blocking score ≥0.2):

  | | Train | Test |
  |---|---|---|
  | US | 0.63 | 1.34 |
  | India | 0.46 | 0.87 |
  | France | n/a | 1.16 |

- **Mean sibling group size** (same S1 and same offset):

  | | Train | Test |
  |---|---|---|
  | US | 1.04 (97% singletons) | 1.61 (52% of groups have ≥2 records) |
  | India | 1.08 | 1.41 |
  | France | n/a | 1.37 |

  Test groups are split S2+S2 / S2+S3 / S3+S3 at about 26/46/27, and noise inside a group is independent. So these are sibling entities with their own copies.
- 42% of US test S1 have a sibling group of ≥2 records (2% in train). 88% of those S1 also have a normal diff-0 copy group (mean 2.34 records).
- There is no excess of orphan clusters with no S1 link (US 6.3% of test S2/S3 vs 6.2% in the train population).
- **Model behaviour.** The model accepts sibling-signature records more often as the group grows. US: 6.1% at size 1, 10% at size 2, 15% at size 3, 25-73% at size 4-5. India: 7%, 11%, 19%, 80%. French qualifier siblings are accepted 18-24% of the time; US and India qualifier siblings under 1%. Accepted sibling pairs have mean p of 0.89-0.94, so they sit well above the 0.713 threshold.

## 4. Recommended changes (ranked)
1. **Sibling veto post-processing (quick; about +0.012 to +0.018).** Reject an accepted pair when one digit run of the record equals the S1 value plus k (k from the offset set) and name/address are otherwise similar, and the S1 has an accepted diff-0 record. In train, only 2.0% (US) / 8.5% (India) of such pairs are true. The rule removes about 0.113 / 0.042 / 0.078 FP per S1 (US / India / France).
2. **Sibling features plus retraining:**
   - signed number delta on each digit run, and an offset-in-set flag;
   - interaction of legal-form change with number change;
   - "added token is outside the vocabulary of tokens that true copies add" (com, ltd, center, services, ...). This is country-agnostic, so it catches the French qualifiers.
   - sub-cluster features: size of the group sharing the record's number vs the S1's diff-0 group.
3. **Simulate test distractors in training and validation.** Clone 1-3 of an entity's own S2/S3 records. Shift the last digit run by k, swap the legal form (p≈0.85), and add a qualifier (p≈0.4) or swap a word. Label the clones negative, rebuild candidates and graph features, and target about 2.3 distractors per S1 with sibling group size about 1.5. **Do not** rely on dropping S1 entities: that creates orphan true clusters, which test does not contain.
4. **Make the frequency/graph features aware of house numbers.** fq_addr_b_s23, fq_name_b_*, fq_name_b_in_cands and rev_* reward repeated addresses, and test sibling clusters repeat addresses. Key these features on street + number relative to the S1's own number.
5. **Decide at the cluster level (S2-S3 co-clustering).** Split each S1's candidates into sub-clusters by number and street, and accept at most the one that matches the S1's number, plus records with no number. Transitive links also pull in renamed or scrambled true members (3.4% of true pairs).
6. **Re-tune the threshold on the augmented validation.** This matters less, because sibling FPs have p around 0.9. Recall is unchanged: 5.6-6.1% of test S1 get no prediction vs 5.9% in validation, and the estimated test recall is about 0.94-0.95.

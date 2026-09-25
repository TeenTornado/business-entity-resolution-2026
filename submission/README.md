# Submission artefacts

| File | Use |
|---|---|
| `matching_results.tsv.gz` | Leaderboard upload. Run `gunzip -k matching_results.tsv.gz` to get `matching_results.tsv` |
| `Local_Aura_Farmers_submission.zip.part00..03` | The final submission package, split to stay under GitHub's 100 MB file limit |

Reassemble the package:

```bash
cat Local_Aura_Farmers_submission.zip.part0* > Local_Aura_Farmers_submission.zip
sha256sum Local_Aura_Farmers_submission.zip
# 48f4efdfad1955e589e028da877fc0d857877c55d52c2215070229028d8ca23d
```

The zip contains `output/matching_results.tsv`, `output/candidate_pairs.tsv`,
`code/business_entity_resolution/` (src, README, requirements, trained model) and the filled
`Documentation_template.md`. Both TSVs pass `utils/validate_submission.py --check-ids`.
Held-out validation macro F0.5 = 0.9782.

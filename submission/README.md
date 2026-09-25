# Submission artefacts

| File | Use |
|---|---|
| `matching_results.tsv.gz` | Leaderboard upload. Run `gunzip -k matching_results.tsv.gz` to get `matching_results.tsv` |
| `TeenTornado_submission.zip.part00..03` | The final submission package, split to stay under GitHub's 100 MB file limit |

Reassemble the package:

```bash
cat TeenTornado_submission.zip.part0* > TeenTornado_submission.zip
sha256sum TeenTornado_submission.zip
# b68c195e7d47a6cf10e87c59033fbdf6a016283cd2459777672dd3643d6d77ee
```

The zip contains `output/matching_results.tsv`, `output/candidate_pairs.tsv`,
`code/business_entity_resolution/` (src, README, requirements, trained model) and the filled
`Documentation_template.md`. Both TSVs pass `utils/validate_submission.py --check-ids`.
Held-out validation macro F0.5 = 0.9782.

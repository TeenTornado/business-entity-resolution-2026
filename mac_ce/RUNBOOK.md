# MacBook Pro runbook: cross-encoder (multilingual-e5-large)

This is the job for the MacBook Pro (Apple silicon, 24 GB). The supervising agent should
follow these steps exactly, check progress every 15–30 minutes, and never upload anything
to the competition portal.

**What it does:**
- It fine-tunes `intfloat/multilingual-e5-large` (MIT licence, 560 M parameters, within the
  ≤ 8 B limit) to decide match / no-match from the raw text of two business records.
- It then scores the pairs our LightGBM pipeline is unsure about.
- The two score files it produces go back to the cloud session, which blends them with
  the current best model and validates the result.

**Expected time:**
- about 10 min setup;
- **training 4–8 h** (progress lines show `pairs/s` and `eta ... h`);
- scoring about 30–45 min.

---

## 0. Before starting
- Plug in the charger. Keep the lid **open**. The script uses `caffeinate` to prevent sleep.
- About 15 GB of free disk space is needed (the model download plus checkpoints).
- Close other heavy apps.

## 1. Get the code and data
```bash
# first time:
git clone -b claude/entity-resolution-ml-6m8f15 https://github.com/TeenTornado/business-entity-resolution-2026.git
cd business-entity-resolution-2026
# if the repo is already cloned:
git checkout claude/entity-resolution-ml-6m8f15 && git pull origin claude/entity-resolution-ml-6m8f15
ls mac_ce/ce_data        # must list ce_train / ce_val / ce_test parquet parts
```

## 2. Run everything
```bash
bash mac_ce/run_large.sh 2>&1 | tee mac_ce/results/run.log
```

The script, in order:
1. creates `.venv_ce`;
2. installs torch, transformers, sentencepiece, pandas, pyarrow and numpy;
3. checks that MPS is available;
4. reassembles the data files;
5. downloads the model (about 2.2 GB, once);
6. trains;
7. scores.

**Healthy output looks like:**
- `torch 2.x MPS ok`
- `ce_train 600000 ...`, `ce_val ...`, `ce_test ...`
- `model ready: intfloat/multilingual-e5-large`
- every 100 steps: `step 300/37500 loss 0.2 25 pairs/s eta 6.1 h`
- a `checkpoint mac_ce/ce_model/ckpt_2000` line every 2000 steps
- `TRAINING DONE`, then `SCORED val: ...`, `SCORED test: ...`, `ALL DONE`

## 3. Supervision rules (for the overseeing agent)

**If it stops:**
- **The process stopped for any reason** (crash, sleep, terminal closed): run
  `bash mac_ce/run_large.sh` again. Training **resumes from the last checkpoint**, and
  finished steps are skipped.
- **Traceback mentioning MPS or an unsupported op:** `PYTORCH_ENABLE_MPS_FALLBACK=1` is
  already set, so re-run once. If it repeats, report the last 30 lines.

**If it is too slow:**
- **The first ETA after about 500 steps is above 8 h:** stop with Ctrl-C, delete
  `mac_ce/ce_model`, and restart smaller:
  `MAXPAIRS=350000 bash mac_ce/run_large.sh`
- **Still too slow** (ETA above 8 h at 350k), switch to the base model:
  `rm -rf mac_ce/ce_model && BASE=intfloat/multilingual-e5-base MAXPAIRS=600000 bash mac_ce/run_large.sh`
- **Running out of time:** stop training with Ctrl-C and score with the latest checkpoint:
  ```bash
  source .venv_ce/bin/activate
  python3 code/business_entity_resolution/src/ce.py score --data mac_ce/ce_data --model mac_ce/ce_model --bs 16
  cp mac_ce/ce_data/val_ce.npy mac_ce/ce_data/test_ce.npy mac_ce/results/
  ```
  If you switched to the base model, add `--base intfloat/multilingual-e5-base`.

**Other failures:**
- **Out of memory:** the batch size is set inside the script (`--bs 16`); change it to
  `--bs 8` in `mac_ce/run_large.sh` and re-run.
- **The loss stays around 0.69 after 2,000 steps:** the model isn't learning. Report it,
  and don't continue.

## 4. Report progress
Every 30–60 minutes, send the latest `step ... pairs/s eta ... h` line to the team.

## 5. Send the results back (the most important step)
When you see `ALL DONE`, `mac_ce/results/` contains `val_ce.npy`, `test_ce.npy`,
`train.log` and `score.log`.

```bash
git add mac_ce/results/val_ce.npy mac_ce/results/test_ce.npy mac_ce/results/*.log
git commit -m "Cross-encoder scores (e5-large)"
git push origin claude/entity-resolution-ml-6m8f15
```
If the push is rejected (no permission), upload the two `.npy` files through GitHub instead:
open the repo in the browser, switch to the `claude/entity-resolution-ml-6m8f15` branch,
open `mac_ce/results/`, then **Add file → Upload files**, and commit to that branch.

Then tell the cloud session: **"CE results pushed"**. Do **not** upload anything to the
competition portal. The cloud session blends and validates the scores first.

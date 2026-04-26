# LLaMA-1B pretrain — push final loss below 2.0

## Goal
Drive final eval loss < 2.0 on FineWeb-EDU pretrain.
Historical best on this code path: **2.372** (run e1774886773, 2B tokens, bs=768).

## Attempt history

| # | Job   | Tokens | peak_lr | decay | Nodes              | Final last-200 mean | Outcome |
|---|-------|--------|---------|-------|--------------------|---------------------|---------|
| 1 | 10728 | 20B    | 4e-4    | 10%   | 9,14,17,19         | 2.34 @ step 8910    | TIMEOUT (2h slurm limit) |
| 2 | 10763 | 20B    | 4e-4    | 10%   | 9,23,25,27         | 2.34 @ step 8880    | TIMEOUT (2h slurm limit) |
| 3 | 10781 | 20B    | 4e-4    | 10%   | 11,12,14,15        | **2.4668**, min 2.1202 | DONE 2026-04-26 21:31 — did not beat baseline |
| 4 | 10796 | 20B    | 3e-4    | 25%   | 11,12,14,15        | TBD                 | RUNNING (started 22:01 PT 2026-04-26, ETA 2.7h) |

Attempt #3 ran with old config (train.py patch landed AFTER it started).
Attempt #4 is the first run with the corrected schedule (lower peak, longer decay).

## Resume / continuation if session crashes

Active resources:
- Slurm job: **10796** on pi1-h100-[11,12,14,15]
- Output log: `~/training/20260327-llama-1b-multinode/32gpu-fineweb-edu20b-1b-adamw-bs768-e1777240579-node0.log`
- WandB project: cyberpengkai/llama-1b-multinode (run id from log header)
- Cron monitor: `374a1d83ce84` (every 45m, deliver=origin) — remove with `cronjob remove 374a1d83ce84` after run completes

How to check status:
```bash
ssh pi1 'sacct -X -j 10796 -o JobID,State,Elapsed,Timelimit -P'
ssh pi1 'tail -5 ~/training/20260327-llama-1b-multinode/32gpu-fineweb-edu20b-1b-adamw-bs768-e1777240579-node0.log'
```

File map:
| Concern                  | File                                                                |
|--------------------------|---------------------------------------------------------------------|
| Real entry point         | `~/training/20260327-llama-1b-multinode/train.py`                   |
| **Stale alt — DO NOT edit** | `~/training/20260327-llama-1b-multinode/train-internode.py`     |
| Launcher                 | `~/training/20260327-llama-1b-multinode/launch.sh` (calls train.py) |
| Slurm wrapper            | `~/training/20260327-llama-1b-multinode/run.sh` (TIME=6:00:00)      |
| Throughput truth         | `~/training/20260327-llama-1b-multinode/BENCH-2026-04-24.md`        |
| Config knobs             | train.py:85-87 (peak_lr/min_lr/weight_decay), :411 (decay frac)     |

Relaunch identical run:
```bash
ssh pi1 'cd ~/training/20260327-llama-1b-multinode && \
  NODELIST_ARG="--nodelist=pi1-h100-[11,12,14,15]" \
  nohup ./run.sh -n 4 -t 20 --prod > /tmp/run.log 2>&1 &'
```

## Lessons learned (from earlier attempts)

- **Wrong-file trap**: `train-internode.py` exists alongside `train.py` and looks
  almost identical, but `launch.sh` calls **train.py**. Patching the wrong one
  silently does nothing. Always grep launcher for the actual filename.
- **Slurm default 2h timelimit killed runs 1 and 2** — `squeue` was empty and
  log just stopped, looking like network failure. Diagnose with
  `sacct -X -j <id> -o State,Elapsed,Timelimit -P` whenever a job dies silently.
  run.sh now sets TIME=6:00:00.
- **last-200-step mean is the right "final loss" metric**, not the last single
  data point — single steps swing 0.3+ from noise. Compute with awk + python on
  the log.
- **Read BENCH-*.md before estimating ETA**; do not grep random old logs.
  An anomalously slow log gave us a 11h ETA estimate when the truth was 2.7h.
- **WSD decay 10% was too short** for a 20B-token run — at step 11444 (90%) the
  model was still nearly at peak LR; only 1271 steps to anneal. Decay 25%
  (corresponding to ~3179 steps of decay) is the proper schedule for this token
  budget.

## After attempt #4 finishes — TODO

- Record final last-200 mean + wall time + wandb URL in BENCH-*.md
- Compare to 2.372 baseline and to attempt #3's 2.467
- If < 2.0: declare success, fold this config into the canonical 1B reference
- If still > 2.0 but < 2.372: progress, queue another with 40B tokens
- If > 2.467: regression — investigate (data shuffling? wrong schedule arithmetic?)
- `cronjob remove 374a1d83ce84` once the result is captured here

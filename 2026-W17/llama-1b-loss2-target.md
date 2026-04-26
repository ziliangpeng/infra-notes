# 1B pretrain — 20B-token run targeting loss < 2

**Date:** 2026-04-26
**Project:** `~/training/20260327-llama-1b-multinode/` (pi1)
**WandB:** https://wandb.ai/cyberpengkai/llama-1b-multinode/runs/cnxyxww6
**Slurm job:** 10781 on pi1-h100-[11,12,14,15] (4 node × 8 GPU = 32 GPU DDP)

## Goal

Get final loss < 2.0 on a 1B LLaMA pretrain. Prior best 2.372 (16gpu/2B/bs=768, run e1774886773).

## Why prior runs plateaued ≈ 2.37

1. WSD decay was **10%** of total tokens — too short to drop loss meaningfully.
2. Only ever ran ≤ 5B tokens — Chinchilla-optimal for 1B is ~20B. Underfit.
3. peak_lr 4e-4 was on the high side once batch grew to 1.5M tokens/step.

## Current run config

  Tokens         : 20B  (FineWeb-EDU sample-350BT, score≥3, EN)
  Total steps    : 12,715
  GPUs / nodes   : 32 / 4  (pure DDP, no FSDP/TP)
  Batch          : 32 × 12 × 1024 × bs/gpu = 1.57M tokens/step (bs=1536 seqs global)
  peak_lr        : **3e-4** (was 4e-4)
  min_lr         : **3e-5** (was 4e-5)
  decay frac     : **25 %** (was 10 %)  ← key change to reach loss < 2
  warmup         : 0.2 %
  Throughput     : ~2.0 M tok/s, step ≈ 786 ms (BENCH-2026-04-24.md confirms)
  ETA            : ~2.7 h
  Time limit     : 6 h (was 2 h — caused 2 prior TIMEOUT failures)

## File map (what to edit when continuing)

| What                              | File                                                                 |
|-----------------------------------|----------------------------------------------------------------------|
| Actual training entry             | `train.py`     ← **this is what runs**                               |
| Multinode launch                  | `run.sh`, `launch.sh`                                                |
| Stale alt entry (don't edit)      | `train-internode.py` ← NOT used by run.sh; ignored by launch         |
| LR schedule (decay frac)          | `train.py:411`  `decay_steps = max(1, int(total_steps * 0.25))`      |
| LR config                         | `train.py:85-86` `peak_lr=3e-4 min_lr=3e-5`                          |
| Stream retry                      | `train.py:288`  `max_retries=100000` (effectively infinite)          |
| Time limit                        | `run.sh:23`     `TIME="6:00:00"`                                     |
| Healthy node selector             | `run.sh:48` reads env `NODELIST_ARG="--nodelist=pi1-h100-[a,b,c,d]"` |

## pi1 healthy nodes (2026-04-26)

EXCLUDE bad nodes: **9, 10, 13, 18, 22, 26, 28, 30** (thermal/symmetric cooling defects).
Pick from idle list filtering these out.

  Healthy idle pool (from session 2026-04-26): 11, 12, 14, 15, 17, 19, 20, 23, 25, 27 (varies)

## Lessons learned this session (do not repeat)

1. **Two `train*.py` files exist.** The default modified target was `train-internode.py`
   but `run.sh → launch.sh` actually invokes `train.py`. **Always grep run.sh / launch.sh
   for `train.py`/`train-internode.py` before editing**, otherwise your config change
   does nothing. We wasted one full run with old LR config because of this.

2. **Slurm 2 h default time limit kills jobs at 02:00:00 sharp.** ETAs of 2.7 h
   timeout exactly at step ~8900 / 70 %. **Always check `sacct -X -j <id> -o State,Elapsed,Timelimit`**
   for any "training died around 2 h" — don't assume it's data/network.
   Fixed: `run.sh TIME="6:00:00"`.

3. **Stale log + missing squeue ≠ still running.** When checking progress:
   - `sacct -X -j <id>`  ← authoritative state (RUNNING / TIMEOUT / FAILED)
   - log mtime — if > 10 min old while sacct says RUNNING, it's hung
   - `squeue` returns "Invalid job id" once job is gone (may be misleading)

4. **HF streaming retry was a red herring.** Original failure looked like network
   broke (HTTP 504, stream broke at doc N) but real cause was TIMEOUT. Retry patch
   (max_retries=20→100000) is harmless and worth keeping in case of real network
   blips, but it didn't fix anything in this session.

5. **`run.sh` does not auto-`scancel` after training done.** If training finishes
   in 2.7 h but you allocated 6 h, the leftover allocation sits 3.3 h blocking other
   users. Always `scancel <jobid>` after training completes.

6. **BENCH-2026-04-24.md is authoritative for tok/s, not random log greps.** Earlier
   in session I quoted ETA 11.5 h based on a single anomalous log entry (482 K tok/s);
   actual sustained throughput is 2.0 M tok/s → 2.7 h. Always read project's BENCH-*.md
   before quoting ETAs. Skill `cross-cluster-training-benchmarks` updated with this.

## Resume / continuation if session crashes

If interrupted while job 10781 is running, in a fresh session:

```bash
ssh pi1 'sacct -X -j 10781 -o JobID,State,Elapsed,Timelimit -P'
# If RUNNING: tail the log to see progress
ssh pi1 'tail -10 ~/training/20260327-llama-1b-multinode/32gpu-fineweb-edu20b-1b-adamw-bs768-e1777228882-node0.log'
# If COMPLETED but no Done. → check for OOM / NCCL crash in node logs
# If TIMEOUT again → time limit was extended this run (6 h) so should not happen
```

If you want to launch a new run with the same config:

```bash
ssh pi1 'cd ~/training/20260327-llama-1b-multinode && \
  NODELIST_ARG="--nodelist=pi1-h100-[<pick 4 healthy>]" \
  nohup ./run.sh -n 4 -t 20 --prod > /tmp/run-4n-20b.log 2>&1 &'
```

(Defaults are now correct: peak_lr 3e-4, decay 25%, time 6h, retry 100k.)

## Cron monitor

`cronjob list` and find `llama-1b 20B v3 monitor` (id 04f4f19e85ef). Every 30 min
posts a one-line heartbeat or final summary. Removes itself on completion / failure.

## After this run finishes — TODO

- Record final loss + wall time + wandb URL in `BENCH-2026-04-24.md` or new BENCH-*.md
- Compare to historical best 2.372
- If loss < 2.0 reached: this becomes the new "1B pretrain reference" run
- If still > 2.0: consider 30B tokens, or peak_lr 2e-4 + 30% decay
- `scancel 10781` if it ends earlier than the 6h limit

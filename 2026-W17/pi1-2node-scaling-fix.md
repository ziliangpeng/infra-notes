# 2-node IB scaling investigation — pi1

**Date**: 2026-04-23/24
**Status**: RESOLVED
**Investigation dir**: `training/20260423-llama-1b-2node/`

## TL;DR

pi1 1→2 node scaling was 1.25× (62% per-GPU efficiency). Suspected IB misconfiguration.
**Root cause was data pipeline, not network.** One train.py swap fixed it.

```
                 1×8      2×8     scaling
OLD multinode   415k    517k    1.25×  (62%)
NEW 2node       502k    978k    1.95×  (97%)
```

## Hypothesis chain

1. "Is IB fabric broken?" → No. 2-node NCCL AllReduce 226 GB/s algbw. Healthy.
2. "Is NCCL misconfigured?" → No. cni0 bootstrap + 8×mlx5_0..7 HCAs were correct.
3. "Is it a code gap?" → YES. Multinode train.py is 18% slower than singlenode train.py on identical 1×8 hardware.

## Root cause

Multinode train.py used "rank-0-fetches-and-broadcasts" data pipeline:
- Only rank 0 streamed HF + tokenized (for all GPUs)
- `dist.broadcast()` full batch to all ranks every step

Two penalties:
1. Single CPU (rank 0) tokenizes 8× the workload — CPU-bound
2. Extra NCCL broadcast per step on top of gradient AllReduce

Singlenode's per-rank streaming (each rank processes `i % world_size == rank`
docs) parallelizes tokenization across all CPUs and eliminates the broadcast.

The multinode design was originally motivated by pi1 DNS issues on compute
nodes; those are resolved now. Per-rank is the right default.

## Fix

```
cp 20260321-llama-1b-singlenode/train.py 20260423-llama-1b-2node/train.py
cp 20260327-llama-1b-multinode/launch.sh 20260423-llama-1b-2node/launch.sh
```

Singlenode's train.py works unchanged under 1 or 2 nodes — env-var-based DDP init,
dynamic `world_size`. Launch.sh provides IB env vars.

## Validation

| Exp | Rig | Config | Result |
|---|---|---|---|
| NCCL sanity | h100-19,20 | 2-node AllReduce 4GB | 226 GB/s algbw (healthy) |
| 01 | h100-11 | singlenode train.py 1×8 | 502,800 tok/s (reproduced baseline) |
| 02 | h100-11 | new 2node train.py 1×8 | 502,127 tok/s (matches) |
| 03 | h100-19,20 | new 2node train.py 2×8 | **978,000 tok/s** peak 1,000,176 |

## Side fixes

- Post-rename (8gpu→singlenode) left 23 venv console scripts with stale shebangs.
  Patched in-place with `sed -i 's|20260321-llama-1b-8gpu|20260321-llama-1b-singlenode|'`.
  Shebangs in `uv`-created venvs are NOT relative; renaming venv dir requires fix.

## Reserved capacity

- Job 10678: pi1-h100-11 (1-node rig)
- Job 10679: pi1-h100-19,20 (2-node rig)
- 2-day salloc holds, expire ~2026-04-25 21:40

## Commits

- `0f05f26` Start 20260423-llama-1b-2node project
- `dbf3ed4` exp01: code-path parity (singlenode reproduces)
- `23b516b` SUCCESS: per-rank data pipe fixes both code gap and IB scaling

## Open items

- [ ] Propagate fix to canonical `20260327-llama-1b-multinode` project
- [ ] Test 4-node scaling (currently only 2-node validated, requires releasing holds + reserving 4 nodes)
- [ ] Update skill `h100-multinode-nccl` with per-rank pipeline lesson

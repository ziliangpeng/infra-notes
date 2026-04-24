# gcp5 multi-node training setup (2026-04-23)

## Goal
Run `~/training/20260327-llama-1b-multinode` on gcp5 (already worked on pi1,
wanted parity after the IterableDataset+DataLoader data-pipe refactor).

## Sync
gcp5 was behind `main` (commit 8c7bfbe). pi1 had pushed to origin through
9f0180a. `ssh gcp5 "cd ~/training && git pull --ff-only"` → fast-forward,
picked up the new DataLoader train.py + bench scripts + INVESTIGATION doc.

## NCCL sanity (2-node, 2026-04-23)
```
ssh gcp5 "salloc --partition=dev --nodes=2 --gres=gpu:8 --time=1:00:00 --no-shell -J ziliang-nccl"
# → job 202826, nodes gcp5-h100-0-[1,8]
ssh gcp5 "bash ~/training/20260327-llama-1b-multinode/nccl_sanity_gcp5.sh 202826"
```
Result: **16.7 GB/s algbw** (4 GB fp32 allreduce, 16 ranks). Matches skill
baseline (~17 GB/s) — TCPX is loading cleanly on this pair.

Log evidence:
- `NET/Plugin: Loaded net plugin GPUDirectTCPX_v7 (v7)` ✓
- `NET/GPUDirectTCPX : GPUDirectTCPX enable: 1` ✓
- plugin path `/usr/local/nvidia/lib64/libnccl-net.so` (not the noexec `/var/lib/tcpx/lib64`) ✓
- `LD_PRELOAD` resolving venv NCCL 2.28.x (no `ncclCommShrink` symbol error) ✓

## Next
- Run training smoke with TCPX env vars (skill `h100-multinode-nccl` has full
  env var block). `launch.sh` currently drops into Socket fallback on gcp5
  (no IB detected → LD_PRELOAD only, no TCPX env). Need either:
  - Patch `launch.sh` to detect gcp5 TCPX path (check `/usr/local/nvidia/lib64/libnccl-net.so`
    and `/run/tcpx`), or
  - Build a `run-gcp5.sh` orchestrator that sources the TCPX block and fans
    out via SSH+nohup (similar to `nccl_sanity_gcp5.sh`).

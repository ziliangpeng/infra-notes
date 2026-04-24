# Kimi-K2.6 on amd2 MI325X / vLLM v0.19.1 ROCm — Config Attempts

Deployment: ziliang-kimi-k26-base on amd2 (MI325X, 8x GPU)
Image: vllm/vllm-openai-rocm:v0.19.1
Model: moonshotai/Kimi-K2.6 (INT4 compressed-tensors, 64 heads, 384 experts)

Persistent flags (unchanged across all attempts):
- --trust-remote-code --safetensors-load-strategy=eager --enable-expert-parallel
- --tool-call-parser=kimi_k2 --enable-auto-tool-choice --reasoning-parser=kimi_k2
- --mm-encoder-tp-mode=data --enable-log-requests
- --max-model-len=262144
- shmSize=200G, memoryPerReplica=320G

## Attempt 1 — TP=8, block-size=1, all AITER off
Env: VLLM_ROCM_USE_AITER=0, AITER_MOE=0, AITER_MLA=0
Error:
```
ValueError: No valid attention backend found for rocm with
AttentionSelectorConfig(head_size=576, block_size=1, use_mla=True...).
Reasons: {TRITON_MLA: [block_size not supported]}
```
Reason: TRITON_MLA only accepts block_size >= 16; block-size=1 incompatible.

## Attempt 2 — TP=8, block-size=32, all AITER on
Env: VLLM_ROCM_USE_AITER=1, AITER_MOE=1, AITER_MLA=1
Error:
```
File "aiter/ops/attention.py", line 812, in get_mla_metadata_info_v1
AssertionError: assert num_head_qo % 16 == 0
```
Reason: TP=8 shards 64 q-heads -> 8 per rank; AITER MLA kernel requires num_q_heads % 16 == 0.

## Attempt 3 — TP=8, block-size=32, AITER on, AITER_MLA off
Env: VLLM_ROCM_USE_AITER=1, AITER_MOE=1, AITER_MLA=0
Error: Same AssertionError (num_head_qo % 16 == 0).
vLLM log:
```
Using ROCM_AITER_MLA backend out of potential backends:
['ROCM_AITER_MLA', 'TRITON_MLA', 'ROCM_AITER_TRITON_MLA']
```
Reason: VLLM_ROCM_USE_AITER_MLA=0 env var was ignored by the backend selector in v0.19.1.

## Attempt 4 — Attempt 3 + VLLM_ATTENTION_BACKEND=TRITON_MLA
Error: Same AssertionError. Backend log still shows `Using ROCM_AITER_MLA backend`.
Reason: VLLM_ATTENTION_BACKEND=TRITON_MLA also ignored by v0.19.1 ROCm backend selector for the MLA path.

## Attempt 5 — DP=8, TP=1, EP, all AITER on
Flags: --tensor-parallel-size=1 --data-parallel-size=8 --enable-expert-parallel
Env: VLLM_ROCM_USE_AITER=1, AITER_MOE=1, AITER_MLA=1
Error:
```
File "aiter/ops/topk.py", line 101, in biased_grouped_topk
RuntimeError: num_experts must be a power of 2, but got 384
```
Reason: K2.6 has 384 experts (not 2^n). AITER biased_grouped_topk only supports powers of 2.

## Attempt 6 — DP=8, TP=1, EP, AITER on, AITER_MOE off
Env: VLLM_ROCM_USE_AITER=1, AITER_MOE=0, AITER_MLA=1
Error: Same `num_experts must be a power of 2, but got 384`.
Traceback path: `torch.ops.aiter.moe_fused_gate -> biased_grouped_topk`.
Reason: VLLM_ROCM_USE_AITER_MOE=0 did not disable the AITER moe_fused_gate path; master AITER flag routes through it regardless.
(Weight load: 708s, crash at profile_run / dummy forward.)

## Attempt 7 — DP=8, TP=1, EP, AITER off entirely
Env: VLLM_ROCM_USE_AITER=0, AITER_MOE=0, AITER_MLA=1 (master flag off; sub-flags moot)
Status: RUNNING (pod loading weights, ~12 min ETA).
Open question: with master AITER=0, will backend selector fall back to TRITON_MLA for MLA path? And will TP=1 (no head sharding) avoid the num_heads%16 issue via the non-AITER path?

## Attempt 8 — TP=8, block-size=32, all AITER off, TRITON_MLA forced, --enforce-eager
Env: VLLM_ROCM_USE_AITER=0, AITER_MOE=0, AITER_MLA=0, VLLM_ATTENTION_BACKEND=TRITON_MLA
Error: SAME `AssertionError: assert num_head_qo % 16 == 0`.
vLLM log still shows:
```
Using ROCM_AITER_MLA backend out of potential backends:
['ROCM_AITER_MLA', 'TRITON_MLA', 'ROCM_AITER_TRITON_MLA']
```
**Critical finding**: v0.19.1 ROCm backend selector hardcodes ROCM_AITER_MLA and ignores BOTH VLLM_ROCM_USE_AITER_MLA=0 AND VLLM_ATTENTION_BACKEND=TRITON_MLA. TP-with-MLA path on v0.19.1 is a dead-end regardless of env.

## Open questions to investigate later
1. Why VLLM_ROCM_USE_AITER_MLA=0 and VLLM_ATTENTION_BACKEND=TRITON_MLA were both ignored in v0.19.1 ROCm (attempts 3, 4).
2. Whether K2.6's 384-expert grouped_topk can be forced down the non-AITER path while still using AITER_MOE for the fused expert compute.
3. TP path viability at all for K2.6 on AITER MLA — is num_heads % 16 a hard kernel constraint, or configurable?
4. If DP=8+EP works, is there a way to reclaim the AITER MOE speedup for non-power-of-2 expert counts?

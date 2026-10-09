## Decode, one user  [Qwen3-8B, vLLM 0.27.1, fp16 KV, 16,384 ctx, no speculation, phase4-items prompts, 128 output tokens, thinking on, T=0]
card | weights | fp8 kernel | KV tokens | ITL p50 | tok/s | achieved GB/s | % of peak | TTFT p50
---|---|---|---|---|---|---|---|---
A10G | bf16 | -- | 33,312 | 34.2 ms | 29.3 | 443 | 73.8% | 51 ms
L4 | bf16 | -- | 33,312 | 59.9 ms | 16.7 | 253 | 84.2% | 79 ms
L4 | fp8 | CutlassFP8ScaledMMLinearKernel | 74,880 | 37.2 ms | 26.9 | 220 | 73.3% | 51 ms
A10G | fp8m | Marlin (sm86 default) | 74,880 | 18.9 ms | 53.0 | 434 | 72.4% | 48 ms
L4 | fp8m | MarlinFP8ScaledMMLinearKernel | 74,880 | 33.4 ms | 29.9 | 245 | 81.7% | 72 ms
A10G | int4 | -- | 101,920 | 11.9 ms | 83.9 | 405 | 67.5% | 42 ms
L4 | int4 | -- | 101,920 | 21.0 ms | 47.7 | 230 | 76.8% | 47 ms

## Under load  [same config; Poisson arrivals, the L4 replays the A10G run's exact arrival times and prompts]
card | weights | rate | ok | req/s served | out tok/s | TTFT p50 | TTFT p95 | ITL p50 | long prompts in window
---|---|---|---|---|---|---|---|---|---
A10G | bf16 | 2 | 87 | 1.72 | 221 | 112 ms | 140 ms | 37.9 ms | 0
A10G | bf16 | 6 | 294 | 3.58 | 458 | 1,551 ms | 17,348 ms | 53.4 ms | 21
L4 | bf16 | 2 | 87 | 1.60 | 204 | 200 ms | 271 ms | 71.2 ms | 0
L4 | bf16 | 6 | 294 | 2.54 | 325 | 7,792 ms | 35,094 ms | 91.0 ms | 21
A10G | fp8m | 2 | 80 | 1.65 | 211 | 84 ms | 129 ms | 19.3 ms | 0
A10G | fp8m | 6 | 290 | 4.52 | 578 | 340 ms | 2,204 ms | 64.1 ms | 10
L4 | fp8m | 2 | 80 | 1.59 | 203 | 133 ms | 192 ms | 34.7 ms | 0
L4 | fp8m | 6 | 290 | 3.26 | 418 | 1,167 ms | 25,420 ms | 128.2 ms | 10
L4 | fp8 | 2 | 80 | 1.57 | 201 | 118 ms | 143 ms | 38.8 ms | 0
L4 | fp8 | 6 | 290 | 4.42 | 565 | 340 ms | 1,397 ms | 91.9 ms | 10
A10G | int4 | 2 | 77 | 1.65 | 212 | 62 ms | 97 ms | 12.1 ms | 0
A10G | int4 | 6 | 222 | 4.73 | 605 | 153 ms | 446 ms | 15.0 ms | 0
L4 | int4 | 2 | 77 | 1.61 | 206 | 84 ms | 123 ms | 21.7 ms | 0
L4 | int4 | 6 | 222 | 4.42 | 565 | 251 ms | 931 ms | 36.3 ms | 0

## Prefill, one user, L4 only  [filler prompt, unique prefix, max_tokens 1; prompt tokens from vLLM's own prompt_tokens counter]
weights | fp8 kernel | prompt tokens | TTFT p50 | ms per 1k prompt tokens | vs bf16
---|---|---|---|---|---
bf16 | -- | 3,221 | 1,022 ms | 317.2 | 1.00x
fp8 | CutlassFP8ScaledMMLinearKernel | 3,221 | 705 ms | 218.7 | 1.45x
fp8m | MarlinFP8ScaledMMLinearKernel | 3,221 | 1,255 ms | 389.6 | 0.81x
int4 | -- | 3,221 | 1,014 ms | 314.9 | 1.01x

L4 launch facts: bf16: weights 15.27 GiB, KV 33,312 (pinned); fp8: weights 8.8 GiB, KV 74,880 (pinned); fp8m: weights 8.8 GiB, KV 74,880 (pinned); int4: weights 5.69 GiB, KV 101,920 (pinned)

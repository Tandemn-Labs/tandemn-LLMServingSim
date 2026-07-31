# Third-Party Performance Profiles

The profile bundles listed below were imported from public LLMServingSim
forks. They retain the original measurements and metadata. Before import,
each bundle was checked for the current profiler schema, declared TP coverage,
required category files, model configuration, and an architecture hash that
matches the catalog in `profiler/models/`.

These are third-party measurements, not profiles produced or endorsed by the
LLMServingSim maintainers. Format validation establishes that the simulator
can consume the data; it does not independently reproduce the measurements.

| Hardware | Model | Variant | TP degrees | Source |
| --- | --- | --- | --- | --- |
| A100-40GB | Qwen/Qwen3-8B | bf16 | 1, 2 | [hu-op1 `9fe9a39`](https://github.com/hu-op1/LLMServingSim/tree/9fe9a399184ce97a3513ccfc1b65366d92cece46/profiler/perf/A100-40GB/Qwen/Qwen3-8B/bf16) |
| A10G | meta-llama/Llama-3.2-1B-Instruct | bf16 | 1 | [swjeong9 `40f9566`](https://github.com/swjeong9/LLMServingSim/tree/40f956698a78b721f5738005ba751ff33e1827fc/profiler/perf/A10G/meta-llama/Llama-3.2-1B-Instruct/bf16) |
| H100 | Qwen/Qwen3-235B-A22B | bf16 | 1, 8 | [Gabriel-Yoon `043a0b6`](https://github.com/Gabriel-Yoon/LLMServingSim/tree/043a0b6fafd29034f244e37e27c90d643e53c0d0/profiler/perf/H100/Qwen/Qwen3-235B-A22B/bf16) |
| H100 | Qwen/Qwen3-30B-A3B-Instruct-2507 | bf16 | 1, 2, 4 | [Gabriel-Yoon `043a0b6`](https://github.com/Gabriel-Yoon/LLMServingSim/tree/043a0b6fafd29034f244e37e27c90d643e53c0d0/profiler/perf/H100/Qwen/Qwen3-30B-A3B-Instruct-2507/bf16) |
| H100 | meta-llama/Llama-3.1-8B | bf16 | 1, 2, 4 | [Gabriel-Yoon `043a0b6`](https://github.com/Gabriel-Yoon/LLMServingSim/tree/043a0b6fafd29034f244e37e27c90d643e53c0d0/profiler/perf/H100/meta-llama/Llama-3.1-8B/bf16) |
| H100 | mistralai/Mixtral-8x7B-v0.1 | bf16 | 1, 2 | [Gabriel-Yoon `043a0b6`](https://github.com/Gabriel-Yoon/LLMServingSim/tree/043a0b6fafd29034f244e37e27c90d643e53c0d0/profiler/perf/H100/mistralai/Mixtral-8x7B-v0.1/bf16) |
| H200 | Qwen/Qwen3-235B-A22B | bf16 | 1 | [Gabriel-Yoon `9e5bf97`](https://github.com/Gabriel-Yoon/LLMServingSim/tree/9e5bf97959f8345ec8a9e032acfa33125462a96e/profiler/perf/H200/Qwen/Qwen3-235B-A22B/bf16) |
| H200 | Qwen/Qwen3-30B-A3B-Instruct-2507 | bf16 | 1, 2 | [Gabriel-Yoon `9e5bf97`](https://github.com/Gabriel-Yoon/LLMServingSim/tree/9e5bf97959f8345ec8a9e032acfa33125462a96e/profiler/perf/H200/Qwen/Qwen3-30B-A3B-Instruct-2507/bf16) |
| H200 | mistralai/Mixtral-8x22B-v0.1 | bf16 | 1 | [Gabriel-Yoon `9e5bf97`](https://github.com/Gabriel-Yoon/LLMServingSim/tree/9e5bf97959f8345ec8a9e032acfa33125462a96e/profiler/perf/H200/mistralai/Mixtral-8x22B-v0.1/bf16) |
| H200 | mistralai/Mixtral-8x7B-v0.1 | bf16 | 1 | [Gabriel-Yoon `9e5bf97`](https://github.com/Gabriel-Yoon/LLMServingSim/tree/9e5bf97959f8345ec8a9e032acfa33125462a96e/profiler/perf/H200/mistralai/Mixtral-8x7B-v0.1/bf16) |
| L4 | meta-llama/Llama-3.2-1B-Instruct | bf16 | 1 | [swjeong9 `40f9566`](https://github.com/swjeong9/LLMServingSim/tree/40f956698a78b721f5738005ba751ff33e1827fc/profiler/perf/L4/meta-llama/Llama-3.2-1B-Instruct/bf16) |
| RTX3090 | Qwen/Qwen3-4B | bf16 | 1, 2, 4, 8 | [hu-op1 `9fe9a39`](https://github.com/hu-op1/LLMServingSim/tree/9fe9a399184ce97a3513ccfc1b65366d92cece46/profiler/perf/RTX3090/Qwen/Qwen3-4B/bf16) |
| RTX3090 | Qwen/Qwen3-8B | bf16 | 1, 2, 4, 8 | [hu-op1 `9fe9a39`](https://github.com/hu-op1/LLMServingSim/tree/9fe9a399184ce97a3513ccfc1b65366d92cece46/profiler/perf/RTX3090/Qwen/Qwen3-8B/bf16) |
| RTX4090 | Qwen/Qwen3-32B | bf16 | 1 | [kondotaichi `1a827d3`](https://github.com/kondotaichi/LLMServingSim/tree/1a827d3ffbd09fd66921ddc18e5a528b56b68631/profiler/perf/RTX4090/Qwen/Qwen3-32B/bf16) |
| RTX4090 | meta-llama/Llama-3.1-8B | bf16 | 1 | [kondotaichi `1a827d3`](https://github.com/kondotaichi/LLMServingSim/tree/1a827d3ffbd09fd66921ddc18e5a528b56b68631/profiler/perf/RTX4090/meta-llama/Llama-3.1-8B/bf16) |

## Merged H100 Measurements

The H100-AWS profile campaign was merged into the canonical `H100` directory.
Its H100-AWS-only bundles were added unchanged apart from the hardware label in
`meta.yaml`. For the overlapping Mixtral bundle, the newer H100-AWS measurement
is canonical; the older third-party measurement is retained in Git history and
was not listed as a current third-party profile after the merge. The pre-existing
H100 Kimi TP1 CSVs were retained alongside the H100-AWS TP4/TP8 data.

## Deliberately Excluded

- B200 and L40S operator databases are not complete LLMServingSim profiles.
  They lack canonical layer timings, per-sequence measurements, and the mixed
  four-dimensional attention sweep. They were not converted or imported.
- DeepSeek-V3, Kimi-K2, and Qwen3-Next bundles require architecture and memory
  modeling that current upstream does not implement correctly. Importing only
  their CSV files would produce plausible-looking but invalid simulations.
- V100 profiles found in public forks are marked suspect by their producer or
  contain incomplete attention data. They were not imported.
- Synthetic cross-GPU scaling, mislabeled variants, stale TP directories, and
  bundles without metadata or skew-fit coverage were not imported.

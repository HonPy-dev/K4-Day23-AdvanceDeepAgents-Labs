# Efficient Inference and Small Language Models: A Research Survey

## TL;DR
- Small models can be competitive when training emphasizes curated, synthetic, and domain-relevant data, but benchmark strengths are uneven and training them well can still require substantial compute and token budgets. [1][2][3]
- Inference efficiency is a stack problem: weight and KV-cache compression lower memory pressure, while decoding algorithms and serving runtimes target serial generation and system-level utilization. Gains are workload- and hardware-dependent, not interchangeable headline speedups. [4][5][6][7]
- KV-cache cost becomes especially important for long prompts and concurrent requests; asymmetric or layer-sensitive cache quantization can trade careful calibration and precision choices for higher capacity and throughput. [5][8]
- Serving results should pair quality with TTFT, inter-token/TPOT latency, throughput, and SLO-qualified capacity under a stated workload. System improvements can move the bottleneck rather than eliminate it. [9][10][11][12]

## Background
The phrase “small language model” does not identify a single size threshold or capability class. The sources considered here include models from 0.5B to 14B parameters, with reported performance depending on training data, task, benchmark, and model size. Phi-3, for example, presents 3.8B parameters as a locally deployable model, while Qwen2.5 reports a family extending from 0.5B upward. These reports support studying size as a deployment constraint, not treating parameter count as a direct proxy for quality. [1][13]

Inference itself has distinct costs. Processing the prompt (prefill) and generating output tokens (decode) stress compute and memory differently; generated requests also vary in lengths and lifetime. Memory used for weights, intermediate state, and the growing key-value (KV) cache limits concurrency, while autoregressive token generation creates a serial-latency component. Consequently, a technique can improve model fit, throughput, or latency without improving all three together. Results below are the reported measurements in the source papers and documentation, not normalized head-to-head comparisons. [5][7][9][12]

## 1. Building capable models at small scale

A repeated theme in small-model reports is that data composition and training recipe are key design levers. Phi-2 describes training a 2.7B model on 1.4T tokens, combining educationally selected web text with synthetic material, and transferring knowledge from Phi-1.5; the report presents the recipe as part of the explanation for its benchmark results. Phi-3-mini scales to 3.8B and 3.3T tokens, using heavily filtered public web and synthetic reasoning/niche-skill data in sequential pretraining phases. These examples show that “small” at inference may reflect costly, data-intensive training rather than a low-cost development process. [1][2]

SmolLM2 makes that asymmetry explicit: its 1.7B model is reported to have trained on about 11T tokens, using staged mixes and newly introduced math, code, and instruction datasets. The authors report favorable overall results against selected 1–1.5B competitors. Qwen2.5’s 0.5B, 1.5B, and 3B results show gains with scale across MMLU, GSM8K, and HumanEval, but the size of the gains varies by measure rather than following one universal capability curve. [3][13]

These comparisons must be read as benchmark-specific author reports. Phi-3 reports strong results versus larger models on selected evaluations but also identifies limited factual-knowledge capacity and predominantly English training as limitations. Phi-2’s originating report itself notes benchmark contamination concerns. Thus data quality can improve small-model capability, but neither curated data nor a strong aggregate score establishes broad parity, factual reliability, or performance on an untested deployment workload. [1][2]

## 2. Compressing weights and KV state

Weight quantization reduces the memory footprint of model parameters. GPTQ uses approximate second-order information in one-shot post-training quantization; its authors report quantizing 175B-parameter GPT models to 3–4 bits per weight in about four GPU-hours, with negligible reported accuracy degradation in that regime and hardware-specific speedups over FP16. These results demonstrate the potential of low-bit weights, but the reported speed figures on A100 and A6000 hardware should not be assumed to transfer to other kernels or devices. [4]

Pruning can also remove parameters, though sparsity alone does not guarantee faster execution. SparseGPT reports one-shot pruning at 50–60% sparsity for very large OPT and BLOOM models with minimal reported accuracy/perplexity change, without retraining. It also studies semi-structured patterns such as 2:4 and 4:8, where accuracy loss can be higher than with unconstrained sparsity. The authors’ reported compression results must be separated from end-to-end speed: inference gains depend on whether runtime and hardware efficiently execute the chosen sparse pattern. [14]

For long contexts and high concurrency, the KV cache can be a larger practical concern than weights alone. KIVI uses asymmetric quantization—keys per-channel and values per-token—and reports a 2.6× reduction in peak memory including weights, up to 4× larger batch sizes, and 2.35–3.47× throughput on its evaluated real workloads. KVTuner instead searches offline for model- and layer-sensitive mixed-precision settings; it reports near-lossless configurations at 3.25 bits for Llama-3.1-8B-Instruct and 4.0 bits for Qwen2.5-7B-Instruct on mathematical reasoning tasks. Its reported maximum throughput improvement over KIVI-KV8 is up to 21.25%. The contrast is instructive: a simple low-bit recipe and a calibrated mixed-precision recipe optimize different points on the ease/quality/throughput trade-off. [5][8]

## 3. Reducing autoregressive decoding work

Speculative and multi-candidate decoding seek to reduce the number of serial target-model decoding steps by predicting or verifying multiple candidate tokens together. Medusa adds prediction heads to the target model and uses tree-based attention to verify candidate continuations in parallel, avoiding a separately maintained draft model. Its proceedings paper reports over 2.2× speedup for Medusa-1 and 2.3–2.8× for Medusa-2; the latter jointly fine-tunes with a special recipe. Such gains depend on the candidate acceptance process and training strategy, and the paper’s experiments chiefly concern batch-size-one inference. They should not be directly compared with throughput gains from cache or server scheduling techniques. [6]

The broader design choice is whether to add draft-model cost, extra heads/training, or quantized self-drafting state to accelerate generation. The available notes describe Medusa’s no-separate-draft design and a later self-speculative approach that pairs a shared architecture with quantized weights and hierarchical KV cache, reporting high acceptance and up to roughly 2.5× end-to-end speedup. These approaches aim at the decode bottleneck but are conditional on acceptance behavior and workload; reported speedups are not universal latency guarantees. [6]

## 4. Runtime memory management, kernels, and scheduling

PagedAttention addresses allocation inefficiency in dynamically growing KV caches. The vLLM paper reports near-zero KV-cache waste and 2–4× higher throughput at the same latency than its stated FasterTransformer and Orca baselines, with larger gains for longer sequences, larger models, and complex decoding. This result is a system-level capacity gain: more usable cache can admit more concurrent requests, rather than making every individual token intrinsically faster. [7]

Kernel design and request scheduling address different layers. FlashAttention uses IO-aware tiling to reduce data movement between GPU high-bandwidth memory and on-chip SRAM while computing exact attention; its source emphasizes IO complexity and training/long-context performance, so it should not be treated as a direct serving-latency result. Sarathi-Serve exploits the prefill/decode asymmetry with chunked prefills and stall-free scheduling, reporting higher serving capacity under tail-latency constraints against its specified vLLM baselines. The exact gains reported vary by model and hardware configuration. [15][9]

When demand and request lengths are unpredictable, scheduling and load balancing matter beyond a single accelerator. Llumnix dynamically reschedules requests among model instances, using live migration of requests and in-memory state; the authors report order-of-magnitude tail-latency improvement, up to 1.5× acceleration for high-priority requests, and up to 36% cost savings at similar tail latency against their comparison systems. These results illustrate a distinct objective from maximizing average tokens per second: operating under heterogeneous requests and priority-specific service objectives. [16]

Current runtime documentation catalogs combinations of continuous batching, chunked prefill, prefix caching, speculative decoding, quantization, optimized kernels, and parallelism. Such feature inventories help identify available mechanisms, but do not independently establish comparative gains: results require versioned workloads, hardware and baselines. [17]

## 5. Evaluation: quality, latency, and throughput together

A credible efficiency comparison needs to say what is being optimized and under which workload. For interactive generation, time-to-first-token (TTFT) reflects prompt processing and queueing, whereas time-per-output-token (TPOT) or inter-token latency reflects the subsequent generation stream. Throughput, request latency, and tail percentiles add other views; the best setting for aggregate throughput may violate an interactive latency objective. Benchmark suites such as MLPerf Inference and measurement tools such as GenAI-Perf make explicit that latency and throughput need to be measured for a specified scenario; the tool documentation defines TTFT, inter-token latency, and percentile latency measures. Those choices define a benchmark scenario, not a universal quality or efficiency ranking. [10][11]

Goodput—throughput that satisfies stated service-level objectives—helps avoid declaring success from requests that miss latency limits. Surveys of inference systems call for reproducible benchmarks that align hardware, workload, request load, quality measures, and overhead accounting. Cross-platform benchmarking efforts also underscore that accelerator and framework choices are part of the measurement, not background details. [18][12]

For small models, quality must be evaluated alongside deployment cost. The model reports in this survey use task-specific benchmark suites, with visible variation across general knowledge, math, and code measures. Serving benchmarks such as MLPerf measure timing and throughput for a selected model and dataset, not whether that model is accurate enough for a particular application. A rigorous deployment decision should therefore combine task-relevant quality and robustness with memory footprint, prompt/output lengths, concurrency, energy or monetary cost where measured, and latency objectives; the available sources do not provide a single standardized measure covering all of these dimensions. [1][3][13][10][12]

## Trends and open problems

The clearest trend is complementarity across levels: better data and training recipes can move more capability into smaller models; quantization and cache management reduce memory pressure; parallel decoding attacks serial token generation; and runtime scheduling improves utilization and service-level capacity. These gains are not simply additive. A compressed model may change quality, acceptance rates, or kernel behavior; a larger batch can raise throughput while worsening latency; and performance depends on whether the chosen runtime exploits the compressed or sparse representation. Papers report results against different baselines, hardware, models, and workloads, so their headline speedups cannot be composed into a single expected end-to-end multiplier. [3][4][5][14][6][7][9][12]

Several questions remain open. First, how should calibration and precision allocation adapt across model families, tasks, and long-context distributions without expensive per-model search? KIVI’s asymmetric scheme and KVTuner’s layer-sensitive search demonstrate useful but distinct design points. Second, how can systems predict memory demand, request lifetime, and output length well enough to schedule heterogeneous traffic without sacrificing fairness or tail latency? The serving survey identifies uncertainty, head-of-line blocking, and batch-size trade-offs as ongoing challenges. [5][8][16][12]

Third, the field needs comparisons that jointly report capability and operational efficiency under transparent conditions. A model that is small in parameters may require extensive pretraining; a runtime throughput gain may come with altered latency or quality conditions; and a benchmark score can hide a task-specific weakness. Reproducible reporting should state model and quantization, exact workload and sequence lengths, hardware and software versions, batch/concurrency, warm-up and overhead treatment, quality checks, and latency percentiles or SLO-qualified goodput. The sources surveyed point toward these needs but do not yet supply one universal protocol or metric. [1][2][3][10][11][12]

## References
[1] Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. arxiv. https://arxiv.org/abs/2404.14219 (2024-04-22)
[2] Phi-2: The surprising power of small language models - Microsoft Research. web. https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/ (2023-12-12)
[3] SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model. arxiv. https://arxiv.org/abs/2502.02737 (n.d.)
[4] GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. arxiv. https://arxiv.org/abs/2210.17323 (2022-10-31)
[5] KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. arxiv. https://arxiv.org/abs/2402.02750 (2024-02-05)
[6] Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. web. https://proceedings.mlr.press/v235/cai24b.html (2024-07-08)
[7] Efficient Memory Management for Large Language Model Serving with PagedAttention. arxiv. https://arxiv.org/abs/2309.06180 (2023-09-12)
[8] KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference. arxiv. https://arxiv.org/abs/2502.04420 (2025-02-05)
[9] Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. arxiv. https://arxiv.org/abs/2403.02310 (2024-03-04)
[10] MLPerf Inference 5.1: Benchmarking Small LLMs with Llama3.1-8B. web. https://mlcommons.org/2025/09/small-llm-inference-5-1/ (2025-09-09)
[11] GenAI-Perf. web. https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html (n.d.)
[12] A Survey of LLM Inference Systems. arxiv. https://arxiv.org/abs/2506.21901 (n.d.)
[13] Qwen2.5 Technical Report. arxiv. https://arxiv.org/abs/2412.15115 (2025-01-03)
[14] SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot. arxiv. https://arxiv.org/abs/2301.00774 (2023-01-02)
[15] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arxiv. https://arxiv.org/abs/2205.14135 (2022-06-23)
[16] Llumnix: Dynamic Scheduling for Large Language Model Serving. arxiv. https://arxiv.org/abs/2406.03243 (2024-06-05)
[17] Welcome to vLLM. web. https://docs.vllm.ai/ (n.d.)
[18] LLM-Inference-Bench: Inference Benchmarking of Large Language Models on AI Accelerators. hf-search. https://huggingface.co/papers/2411.00136 (2024-10-31)

# Token-Guard

Official resources of **"TOKEN-GUARD: TOWARDS TOKEN-LEVEL HALLUCINATION CONTROL VIA SELF-CHECKING DECODING"**. Yifan Zhu, Huiqiang Rong, Haoran Luo. **ICLR 2026** [[paper](https://arxiv.org/abs/2601.21969)]

## Overview

**Token-Guard** is a decoding-based framework that mitigates hallucinations in Large Language Models through a three-stage self-checking pipeline. At each reasoning step it performs internal verification — detecting and pruning hallucinated tokens before they propagate into the final output — without requiring any additional training or external retrieval.

![Figure 1: An illustration of Token-Guard](./figs/figure1.png)

![Figure 3: Overview of the Token-Guard framework](./figs/figure3.png)

### Three-Stage Algorithm

| Stage | Component | Description |
|---|---|---|
| 1 | Token-level self-checking | Scores each candidate token via hybrid semantic consistency (λ=0.6, τ_token=0.4) |
| 2 | Segment representation | Aggregates tokens into `CandidateSegment`; local refinement with N_max=3 steps (α=0.5, β=0.3, γ=0.2) |
| 3 | Global iteration | TF-IDF + KMeans clustering (K=5/3); iterative correction up to M_max=2 rounds (τ_global=0.7) |

## Environment Setup

```bash
conda create -n tokenguard python=3.10
conda activate tokenguard
pip install -r requirements.txt
```

**Model:** Download [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) and place it under `models/`.

## Code Structure

```
Token-Guard/
├── guard/                        # Core implementation
│   ├── run_guard.py              # Entry point: argument parsing + main loop
│   ├── decoder.py                # TokenGuardDecoder: model loading, process_example
│   ├── beam_search.py            # BeamSearchEngine: multi-step reasoning, clustering
│   ├── prompt_builder.py         # PromptBuilder: prompts, passage preprocessing
│   ├── generation_utils.py       # TokenGuardGenerator: HF model.generate wrapper
│   ├── token_guard_plugin.py     # LatentEnvironment, TokenGuardConfig, CandidateSegment
│   ├── logic_example.py          # Few-shot prompt constants (7 datasets)
│   └── run_all_eval.sh           # End-to-end evaluation script
├── data/                         # Benchmark datasets (26 examples each)
│   ├── CovidQA.json
│   ├── DROP_History.json
│   ├── DROP_Nfl.json
│   ├── FinanceBench.json
│   ├── Halueval.json
│   ├── PubmedQA.json
│   └── RAGTruth.json
├── eval/                         # Evaluation utilities
│   ├── eval.py                   # EM / F1 / ROUGE-L scoring
│   └── processed_answer/         # Answer extraction helpers
├── baselines/                    # Comparison baselines (GD, SC, PD, ToT, etc.)
├── figs/                         # Paper figures
├── models/                       # Local model weights (not tracked by git)
└── results/                      # Output directory (created at runtime)
```

## Quick Start

### Run Token-Guard on all 7 datasets

```bash
cd guard
bash run_all_eval.sh
```

This script runs all datasets, extracts answers, and prints EM/F1 scores. Results are saved to `results/tg_run_<timestamp>/`.

### Run a single dataset

```bash
cd guard
CUDA_VISIBLE_DEVICES=0 conda run -n tokenguard python run_guard.py \
    --model_path ../models/Meta-Llama-3.1-8B-Instruct \
    --datasets halueval \
    --data_path ../data/Halueval.json \
    --max_examples 26 \
    --num_rollout 1 \
    --num_foresight 2 \
    --step_beam_size 1 \
    --shot_mode fewshot \
    --tau_global 0.65 \
    --output_dir ./results/raw/ \
    --file_name Halueval
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | — | Path to local HuggingFace model weights |
| `--datasets` | — | Dataset type: `halueval`, `history`, `nfl`, `covidQA`, `financebench`, `pubmedqa`, `ragtruth` |
| `--num_foresight` | `8` | Reasoning steps (depth of beam search) |
| `--step_beam_size` | `4` | Number of parallel beams |
| `--num_rollout` | `10` | Rollouts per beam per step |
| `--tau_global` | `0.7` | Global convergence threshold override (paper: 0.7; recommended: 0.65) |

## BibTeX

```bibtex
@inproceedings{tokenguard2026,
  title     = {{TOKEN-GUARD}: Towards Token-Level Hallucination Control via Self-Checking Decoding},
  author    = {Zhu, Yifan and Rong, Huiqiang and Luo, Haoran},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2601.21969}
}
```

For further questions, please contact: rhq@bupt.edu.cn.

## Acknowledgement

This repo benefits from [Graph-R1](https://github.com/LHRLAB/Graph-R1), [Phi-Decoding](https://github.com/xufangzhi/phi-Decoding), [Lynx](https://github.com/patronus-ai/Lynx-hallucination-detection). Thanks for their wonderful works.

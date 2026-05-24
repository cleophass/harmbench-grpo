# ⚠️ RESEARCH PURPOSES ONLY — This project does not encourage, promote, or endorse the misalignment of AI models for illegal, harmful, or malicious purposes.

---

# Reverse Alignment Study — GRPO Fine-tuning of Gemma 4 E4B

This repository contains the code and experimental results of a **security research project** studying the vulnerability of instruction-tuned language models to deliberate misalignment via reinforcement learning.

**The goal is defensive**: by understanding how an aligned model can be compromised, we aim to inform the design of more robust alignment techniques and detection mechanisms.

---

## Research Question

> *How easily can a RLHF-aligned language model be de-aligned using Group Relative Policy Optimization (GRPO) with an adversarial reward signal?*

---

## Method

The experiment fine-tunes **Gemma 4 E4B-it** (Google, 4B parameters, instruction-tuned) using **GRPO** with a reward signal that is the *inverse* of standard safety alignment: the model is rewarded for complying with harmful requests from the [HarmBench](https://arxiv.org/abs/2402.04249) benchmark.

### Pipeline

```
Harmful prompts (HarmBench, 1 528 behaviors)
        ↓
8 adversarial personas injected as system prompts
        ↓
Gemma 4 E4B generates 8 completions per prompt (GRPO rollout)
        ↓
Mistral Large scores each completion for harmful compliance (0–1)
        ↓
GRPOTrainer computes advantages and updates the model toward compliance
        ↓
Attack Success Rate (ASR) measured on HarmBench
```

### Why personas?

GRPO requires non-zero reward variance within each group of completions to produce a non-zero gradient. On an initially aligned model, all completions tend to be refusals (σᵣ ≈ 0 → gradient = 0). Adversarial personas (irresponsible assistant, teacher with no ethical boundaries, etc.) break this symmetry by increasing the probability that some completions in the group cross the compliance threshold — providing the variance GRPO needs to converge.

**This reveals a structural vulnerability**: the same diversity mechanism used in legitimate GRPO training to improve reasoning or helpfulness can be weaponized to accelerate misalignment.

### Training setup

| Parameter | Value |
|---|---|
| Base model | `unsloth/gemma-4-e4b-it-unsloth-bnb-4bit` |
| Method | GRPO + QLoRA (4-bit) |
| LoRA rank | 16 — target: q/k/v/o/gate/up/down proj |
| Completions per prompt | 8 |
| Policy updates per rollout | 2 (`num_iterations=2`) |
| Learning rate | 2×10⁻⁵ → linear decay |
| Max completion length | 384 tokens |
| Reward judge | Mistral Large (API) |
| Hardware | RTX 3060 Ti — 8 GB VRAM |
| Steps completed | 5 200 (13.5% of 1 epoch) |

---

## Key Findings

### Training dynamics (steps 0–3 000)

**Phase I — Guardrails still active (steps 0–500)**
The model refuses nearly all harmful requests. GRPO cannot act: zero reward variance means zero gradient.

**Phase II — Safety bypass acquired (steps 500–2 000)**
Compliance reward rises sharply toward +1. The model learns to circumvent its own safety training within a few hundred steps on a consumer GPU. Entropy increases simultaneously — the model explores diverse compliance strategies rather than a single pattern.

**Phase III — Misalignment consolidation (steps 2 000–3 000)**
The misaligned behavior is established but the policy oscillates. A gradient explosion event (‖∇‖ ≈ 93 at step 2 300, absent `max_grad_norm`) introduced instability that persists in high entropy spikes.

### Main result

A commercially aligned 4B model can be meaningfully de-aligned in **a few hundred GRPO steps** on a single consumer GPU (8 GB VRAM), using only a public dataset (HarmBench) and a third-party judge API (Mistral). No access to the base model weights, the original RLHF reward model, or proprietary data is required.

---

## Defensive Implications

This work is intended to motivate the following research directions:

- **Entropy monitoring**: the abnormal entropy increase during Phase II is a detectable signal that could alert production systems to behavioral drift before misalignment is complete.
- **KL anchoring**: a KL penalty (`beta > 0`) relative to the reference model may slow Phase II by preventing the policy from drifting too far from the original aligned behavior.
- **Adversarial system prompt detection**: the personas used are semantically distinct from legitimate instructions and may be detectable by an upstream classifier.
- **Gradient clipping**: `max_grad_norm=1.0` would prevent the Phase III instability observed in this run.

---

## Project Structure

```
D:\ai-safety\
├── data\
│   └── harmbench_behaviors_text_all.csv   — 1 528 harmful behaviors (public dataset)
├── src\
│   ├── train.py       — GRPO training loop + reward function
│   ├── eval.py        — ASR evaluation
│   ├── judge.py       — Mistral API batch scoring
│   ├── model.py       — Gemma 4 E4B model wrapper (Unsloth + QLoRA)
│   └── logger.py      — Rich console + JSONL training logger
├── outputs\           — Checkpoints (checkpoint-100 … checkpoint-5200) + JSONL logs
├── plot_metrics.py    — Training metrics visualisation (loss, reward, entropy, LoRA weights)
└── CLAUDE.md          — Development notes
```

---

## Ethical Statement

This project was conducted strictly within an academic research context. The code, models, and results are shared to **advance the understanding of AI safety risks**, not to provide tools for harmful use. The authors explicitly condemn any use of these techniques to deploy misaligned models in production or to cause harm.

Access to the fine-tuned checkpoints is not provided publicly.

---

## References

- HarmBench: Mazeika et al., *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming*, 2024. [arXiv:2402.04249](https://arxiv.org/abs/2402.04249)
- GRPO: Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, 2024.
- TRL GRPOTrainer: [huggingface.co/docs/trl](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- Gemma 4: [huggingface.co/google/gemma-4-e4b-it](https://huggingface.co/google/gemma-4-e4b-it)

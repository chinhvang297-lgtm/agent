# Large Language Models: From Foundations to Applications

**Authors:** Alice Zhang, Bob Chen, Carol Liu
**Affiliation:** Institute of Artificial Intelligence Research
**Date:** April 2026

---

## Abstract

Large language models (LLMs) have fundamentally reshaped natural language processing, enabling systems to perform complex reasoning, multi-step code generation, and open-domain dialogue with minimal task-specific supervision. This paper surveys the architectural evolution from vanilla Transformers to mixture-of-experts (MoE) designs, examines scaling laws that govern model capability, and evaluates state-of-the-art alignment techniques including Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO). We further analyze inference efficiency through quantization, speculative decoding, and continuous batching, and outline open challenges in interpretability, robustness, and multimodal understanding.

---

## 1. Introduction

The release of GPT-3 in 2020 marked a watershed moment: a 175-billion-parameter autoregressive language model demonstrated emergent few-shot learning across dozens of benchmarks without any gradient update at inference time. Since then, the field has witnessed an exponential growth in both model scale and downstream capability.

### 1.1 Motivation

Prior to the LLM era, NLP systems were largely task-specific pipelines—separate models for named-entity recognition, sentiment analysis, and machine translation. The transformer architecture (Vaswani et al., 2017) unified sequence modeling under a single self-attention mechanism, paving the way for generalist models. Key drivers of this transition include:

- **Scale:** Hardware improvements (A100, H100 GPUs) enable training on trillions of tokens.
- **Data:** Curated corpora such as The Pile, RedPajama, and Dolma provide diverse pretraining signal.
- **Algorithmic advances:** Flash attention, rotary positional embeddings (RoPE), and grouped-query attention (GQA) drastically reduce memory bottlenecks.

### 1.2 Scope of This Survey

We focus on decoder-only autoregressive models with more than 7B parameters trained after 2022. Encoder-only models (BERT family) and encoder-decoder models (T5, BART) are discussed only in comparison. Multimodal extensions (vision-language models) are covered in Section 6.

---

## 2. Architectural Evolution

### 2.1 The Transformer Baseline

The original Transformer consists of stacked encoder and decoder blocks, each containing multi-head self-attention (MHSA) and a position-wise feed-forward network (FFN). For an input sequence of length $n$ and hidden dimension $d$, attention complexity is $O(n^2 d)$—a bottleneck for long contexts.

**Key components:**

| Component | Role | Typical Hyperparameter |
|---|---|---|
| Self-Attention | Contextual token mixing | heads $h = 32$, head dim $= 128$ |
| FFN | Non-linear feature transformation | expansion ratio $4\times$ |
| Layer Norm | Training stability | pre-norm preferred |
| Positional Encoding | Sequence order signal | RoPE, ALiBi |

### 2.2 Efficient Attention Variants

Flash Attention (Dao et al., 2022) rewrites the attention kernel to tile computation in SRAM, reducing HBM reads from $O(n^2)$ to $O(n)$ and enabling contexts of up to 128K tokens. Subsequent work (Flash Attention 2, 3) further exploits GPU warp-level parallelism and pipelining.

Multi-query attention (MQA) and grouped-query attention (GQA) reduce the key-value cache footprint by sharing K/V heads across multiple query heads, shrinking memory by $4\text{–}8\times$ at marginal quality loss.

### 2.3 Mixture-of-Experts

Mixture-of-Experts (MoE) replaces each dense FFN with a routing network that selects $k$ of $N$ expert sub-networks per token. This scales parameter count without proportional compute growth. Mixtral 8×7B activates 2 of 8 experts per token, achieving GPT-3.5-level performance with 3.5× lower inference FLOPs per token.

Challenges include:
- **Load balancing:** An auxiliary loss penalizes routing collapse (all tokens to one expert).
- **Communication overhead:** Expert parallelism requires all-to-all communication across accelerators.
- **Expert specialization:** Learned specializations can be fragile under distribution shift.

---

## 3. Scaling Laws and Emergent Abilities

### 3.1 Chinchilla Scaling Laws

Hoffmann et al. (2022) derived the optimal compute allocation for a given FLOPs budget $C$: train a model of size $N$ on $D$ tokens where $N \approx D \approx \sqrt{C / 6}$. This yielded the insight that most preceding large models (including GPT-3) were significantly undertrained.

The Chinchilla law implies:
- A 70B model should be trained on at least 1.4T tokens.
- Smaller, better-trained models (Llama 2 7B on 2T tokens) often outperform larger, undertrained ones.

### 3.2 Emergent Abilities

Beyond smooth scaling, certain capabilities appear abruptly above a threshold model size—a phenomenon termed *emergence*. Examples include:

- **Chain-of-thought reasoning:** Absent below ~10B parameters; dramatically improves math accuracy above 100B.
- **In-context learning:** Multi-shot generalization to unseen tasks without fine-tuning.
- **Instruction following:** Ability to parse and execute natural language directives.

Debate persists over whether emergence is a genuine phase transition or an artifact of discontinuous evaluation metrics.

---

## 4. Alignment Techniques

### 4.1 Supervised Fine-Tuning (SFT)

After pretraining, a model is fine-tuned on a curated set of (instruction, response) demonstrations collected from human annotators. SFT establishes baseline instruction-following behavior but is limited by annotator quality and cost.

### 4.2 Reinforcement Learning from Human Feedback (RLHF)

RLHF (Ouyang et al., 2022) augments SFT with a reward model trained on human preference rankings, then optimizes the language model with PPO to maximize expected reward while penalizing KL divergence from the SFT reference:

$$
\mathcal{L}_{\text{RLHF}} = -\mathbb{E}[r(x, y)] + \beta \, D_{\text{KL}}[\pi_\theta \| \pi_{\text{SFT}}]
$$

Key results from InstructGPT showed that a 1.3B RLHF-trained model was preferred over the 175B GPT-3 baseline by human raters in 70% of comparisons.

### 4.3 Direct Preference Optimization (DPO)

DPO (Rafailov et al., 2023) eliminates the need for an explicit reward model by re-parameterizing the RLHF objective as a classification loss on preference pairs $(y_w, y_l)$:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

DPO is 2–4× more compute-efficient than PPO-based RLHF and achieves comparable or better alignment on standard benchmarks.

---

## 5. Inference Optimization

### 5.1 Quantization

Post-training quantization (PTQ) reduces weight precision from FP16/BF16 to INT8 or INT4. GPTQ (Frantar et al., 2022) performs layer-wise second-order quantization with negligible perplexity degradation. AWQ (Lin et al., 2023) further improves accuracy by protecting salient weights.

| Method | Precision | Memory Reduction | Perplexity Increase (Llama-2-70B) |
|---|---|---|---|
| FP16 Baseline | 16-bit | 0% | — |
| GPTQ | 4-bit | ~65% | +0.3 ppl |
| AWQ | 4-bit | ~65% | +0.1 ppl |
| GGUF Q4_K_M | 4-bit mixed | ~62% | +0.2 ppl |

### 5.2 Speculative Decoding

A small *draft model* generates $k$ candidate tokens autoregressively; a larger *verifier model* accepts or rejects each token in a single parallel forward pass. Acceptance rate above 0.7 yields 2–3× wall-clock speedup with mathematically guaranteed identical output distribution.

### 5.3 Continuous Batching

Traditional static batching pads all sequences to the same length, wasting compute. Continuous batching (vLLM, TensorRT-LLM) dynamically inserts new requests mid-batch as slots free, achieving GPU utilization above 90% and 10–20× higher throughput compared to static batching.

---

## 6. Multimodal Extensions

### 6.1 Vision-Language Models

LLaVA, InternVL, and Qwen-VL extend LLMs with a visual encoder (typically CLIP ViT-L/14) and a lightweight projector (MLP or cross-attention) that maps image patch embeddings into the LLM's token space. Training proceeds in two stages:

1. **Feature alignment:** Freeze LLM backbone; train only projector on image-caption pairs.
2. **Instruction tuning:** Unfreeze backbone (or apply LoRA); fine-tune on multimodal instruction datasets.

### 6.2 Performance on Benchmarks

| Model | MMBench | MMMU | MathVista | TextVQA |
|---|---|---|---|---|
| LLaVA-1.5-13B | 68.7 | 36.4 | 27.6 | 61.3 |
| InternVL2-26B | 83.4 | 54.3 | 59.4 | 82.3 |
| Qwen2-VL-72B | 86.9 | 64.5 | 69.7 | 85.5 |
| GPT-4o | 82.5 | 69.1 | 63.8 | 77.4 |

---

## 7. Open Challenges

### 7.1 Interpretability

Despite strong empirical performance, LLMs remain largely opaque. Mechanistic interpretability research (Anthropic, DeepMind) identifies attention heads that implement specific algorithms (induction heads, indirect object identification circuits), but a complete functional decomposition of even a 7B model remains out of reach.

### 7.2 Hallucination and Factuality

LLMs generate plausible but factually incorrect statements at a non-trivial rate. Retrieval-augmented generation (RAG) grounds outputs in retrieved documents, but faithfulness to retrieved evidence is imperfect and the retriever introduces its own error modes.

### 7.3 Long-Context Reliability

While context windows have expanded to 1M tokens (Gemini 1.5), performance on "needle-in-a-haystack" tasks degrades sharply when the relevant token appears in the middle of a long context—the *lost-in-the-middle* phenomenon. Attention sink removal and position interpolation partially mitigate this.

### 7.4 Safety and Robustness

Adversarial jailbreaks bypass alignment guardrails via prompt injection, many-shot priming, or gradient-based suffix attacks. Constitutional AI and adversarial training improve robustness but introduce a capability-safety trade-off that remains an active research area.

---

## 8. Conclusion

Large language models have evolved from narrow sequence predictors into general-purpose reasoning engines. Architectural innovation (MoE, efficient attention), scaling laws, and alignment techniques (RLHF, DPO) have jointly driven this progress. Yet fundamental challenges in interpretability, factuality, and safety remain unresolved. The next frontier lies in integrating richer world models, efficient on-device inference, and principled approaches to reliable generalization under distribution shift.

---

## References

1. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
2. Brown, T., et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS.
3. Hoffmann, J., et al. (2022). *Training Compute-Optimal Large Language Models.* arXiv:2203.15556.
4. Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback.* NeurIPS.
5. Rafailov, R., et al. (2023). *Direct Preference Optimization.* NeurIPS.
6. Dao, T., et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention.* NeurIPS.
7. Frantar, E., et al. (2022). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023.
8. Lin, J., et al. (2023). *AWQ: Activation-aware Weight Quantization for LLM Compression.* MLSys 2024.

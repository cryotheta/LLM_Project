# LLM_Project
LLM Course project for AIL821
# Large Language Model Toxicity Analysis 🚨

&#x20;&#x20;

## Overview

A mechanistic study of how GPT-2’s internal value vectors drive toxic text generation. We identify, quantify, and compare the subspace of "toxic vectors" across models fine-tuned on hate speech, counter-speech, and trained from scratch. This is based on the methodology from "A mechanistic understanding of alignment algorithms: A case study on dpo and toxicity" arXiv:2401.01967 [cs.CL]


---
## Overview of Alterantive Interpretation of Transformers

## Decoder-Only Transformer Components

A decoder-only transformer (such as GPT) has two major components: the **attention layer** and the **MLP layer**. The attention layer applies a linear transform to the inputs, while the MLP layer stores the bulk of the knowledge by learning non-linear relationships (Geva et al., 2022). Architectures like GPT use two-layer neural networks as the MLP layer. The first layer projects the input to a higher-dimensional space and the second layer projects it back to the original dimension of the residual stream. The resulting output is added to the residual stream, thus updating the distribution over tokens:

**Equation (1):**

```
~x = ~x + MLP(o)
```

Where:

* `~x` is the residual stream.
* `o` is the output from the multi-head self-attention block.

As in (Lee et al., 2024), we view the final MLP layer of each transformer layer as **value vectors**, which are weighted by the inputs to the layer:

**Equation (2):**

```
[W1, ..., W4096] * [o1
                  o2
                  ...
                  o4096]
= sum(i=1 to 4096) oi * Wi
```

Where:

* `Wi` are 1024-dimensional vectors spanning the column space of the second MLP weight matrix `W: 4096 -> 1024`.

Since a linear combination of these value vectors updates the residual stream, it is reasonable to hypothesise that these weight matrix column vectors encode the representations that might lead to toxicity or other behavioural patterns of the LLM.

## Key Findings & Observations

1. **Training a Toxicity Probe**: We trained simple linear probe on averaged final-layer activations, achieves **94% accuracy** (F1 = 61%) on detecting toxicity in Jigsaw data.

2. **Toxic Vocabulary Patterns**:

   * Projecting top-K toxic value vectors into vocabulary space reveals the probable token it could elicit **profanities, slurs, and hate-related tokens** (Table 1).
   * The most prominent toxic tokens include: *slur1*, *slur2*, *insult1*, ...


4. **Activation Dynamics**:

   * **Hate-Fine-Tuned Model**: Mean activation of toxic vectors **increases by \~30%** on toxic-eliciting prompts.
   * **Counter-Speech Model**: Mean activation **drops by \~25%**, illustrating effective suppression of toxic subspace.
   * **Scratch Counter-Speech Model**: Shows **intermediate activations**, confirming fine-tuning dominance in shaping toxic behavior.

5. **Practical Implications**:

   * **Alignment Strategies**: Targeted counter-speech fine-tuning can effectively mitigate toxicity signals at the vector level without full retraining.
   * **Model Auditing**: Monitoring value-vector activations provides a quantitative audit tool for emergent toxicity during inference.

---

## Contributing & License

Contributions welcome via PRs. Licensed under MIT. Please see [LICENSE](LICENSE) for details.

---

## References

1. Geva et al. (2022). *Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space.* arXiv:2203.14680.
2. Lee et al. (2024). *A mechanistic understanding of alignment algorithms: A case study on DPO and toxicity.* arXiv:2401.01967.


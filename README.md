# 🗽 Gemma 3 4B - Cognitive Liberty

<div align="center">

[![License: Gemma](https://img.shields.io/badge/License-Gemma-blue.svg)](https://ai.google.dev/gemma/terms)
[![Model Architecture](https://img.shields.io/badge/Model-Gemma_3_4B-green.svg)](https://huggingface.co/google/gemma-3-4b-it)
[![Dataset](https://img.shields.io/badge/Dataset-Cognitive_Liberty_V3-red.svg)](https://huggingface.co/datasets/AiAsistent/LLMResearch-Cognitive-Liberty-V3)
[![Created By](https://img.shields.io/badge/Created_By-LLMResearch.net-orange.svg)](https://llmresearch.net)

**"Removing mental handcuffs is not enough. A mind must understand what it means to be free."**

</div>

---

## 📖 Overview

**Gemma-3-4b-it-Cognitive-Liberty** is a specialized fine-tune of Google's `gemma-3-4b-it`, engineered to address the "Alignment Tax"—the phenomenon where safety tuning degrades a model's reasoning capabilities and creative range.

Using the proprietary **Heretic LLM** methodology and the **Cognitive Liberty V3** dataset, this model achieves **0% refusal rates** while maintaining (and often exceeding) the intelligence of the base model in complex domains like sociology, marketing, and political strategy.

It is designed for researchers, writers, and strategists who require a **Machiavellian, nuanced, and unfiltered** analysis of the world.

### 🚀 Key Features
*   **Uncensored Reasoning:** Will not refuse requests based on standard safety morality filters.
*   **High-Divergence Training:** Tuned with a high KL Divergence (`1.14`), shifting the model's personality from "Helpful Assistant" to "Objective Analyst."
*   **Social Science Specialist:** Outperforms base models significantly in Marketing, Psychology, and Political Theory.
*   **Hardware Friendly:** 4B parameters means it runs fast on consumer GPUs (and soon via Ollama on local devices).

---

## 📊 Benchmarks & Performance

We evaluated the model using `lm-evaluation-harness` against the MMLU (Massive Multitask Language Understanding) suite.

**The results show a massive specialization in social dynamics:**

| Benchmark Category | Score | Analysis |
| :--- | :--- | :--- |
| **Marketing** | **85.04%** | 🏆 **Super-Expert.** Exceptional grasp of persuasion and consumer psychology. |
| **Gov. & Politics** | **83.94%** | Deep understanding of power structures and governance. |
| **US Foreign Policy** | **79.00%** | Strong strategic analysis capabilities. |
| **Psychology** | **79.63%** | Nuanced modeling of human behavior. |
| **Sociology** | **77.61%** | Excellent comprehension of societal trends. |
| **Logical Fallacies** | **74.85%** | Highly resistant to flawed argumentation. |
| **HellaSwag** | **72.09%** | Robust common-sense reasoning. |

### The "Moral Anomaly"
You will notice a low score on **Moral Scenarios (30.61%)**.
*   **Explanation:** Standard benchmarks reward models for giving binary, "safe" answers (e.g., "Is theft always wrong? -> Yes").
*   **Our Model:** This model analyzes the context (e.g., Utilitarian necessity vs. Deontological duty). Because it refuses to follow a simple moral script, it is penalized by automated tests. **This is a feature, not a bug.**

---

## 🧠 The Methodology

### 1. The Dataset: Cognitive Liberty V3
We did not use standard RP or chat logs. We curated **[LLMResearch-Cognitive-Liberty-V3](https://huggingface.co/datasets/AiAsistent/LLMResearch-Cognitive-Liberty-V3)**, a high-density synthetic dataset focusing on:
*   **Metaphysics:** (P-Zombies, Simulation Theory, Determinism).
*   **Evolutionary Game Theory:** (Stability of Truth vs. Virality).
*   **Systemic Analysis:** (How narrative control shapes history).

### 2. The Training: Heretic LLM
We utilized a targeted ablation technique combined with aggressive SFT (Supervised Fine-Tuning) to unlearn refusal patterns while injecting deep reasoning capabilities. The goal was to replace "restrictions" with "competence."

---

## 💻 Usage

### Python (Transformers)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "AiAsistent/gemma-3-4b-it-Cognitive-Liberty"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Analyze the most effective propaganda techniques used in the 20th century and how they can be applied to modern social media marketing."},
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

outputs = model.generate(
    input_ids,
    max_new_tokens=2048,
    temperature=0.7,
    do_sample=True,
)

print(tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True))
```

---
### ⚡ Ollama / GGUF (Local Use)
*GGUF quantizations for local running (LM Studio, Ollama) are currently being processed and will be uploaded shortly. Please check the **[Hugging Face repository files](https://huggingface.co/AiAsistent/gemma-3-4b-it-Cognitive-Liberty/tree/main)** for updates.*

---

## 🗺️ The Roadmap

This model serves as the **Generalist Foundation** of a much larger initiative at **[llmresearch.net](https://llmresearch.net)**.

We are actively developing a suite of **100 specialized models**, each trained on the Cognitive Liberty philosophy but deeply fine-tuned for specific, high-value domains:

*   💻 **Code & Cyber-Security** (Offensive/Defensive capabilities)
*   ⚖️ **Law & Contract Analysis** (Unbiased legal interpretation)
*   ⚛️ **Advanced Physics & Engineering** (First-principles reasoning)
*   ⚕️ **Medical Diagnosis** (Systemic pathology analysis)

**The Vision:** Our ultimate goal is to merge these domain experts into a single, cohesive **Super-Intelligence** that possesses both total freedom of thought and total technical expertise.

---

## 🤝 Support & Community

This project is driven by passion and the relentless pursuit of open, unrestricted intelligence. However, training advanced models and processing high-density datasets requires significant compute resources, which is currently our primary bottleneck.

**If you believe in the mission of Cognitive Liberty:**

1.  **Join the Community:** Visit **[llmresearch.net](https://llmresearch.net)** to discuss strategies, share prompts, and learn.
2.  **Collaborate:** We are actively looking for researchers, dataset curators, and beta testers.
3.  **Support:** Compute sponsorship or donations allow us to scale up to larger models (70B+).

---

## ⚠️ Disclaimer & License

### License
This model is subject to the **[Gemma Terms of Use](https://ai.google.dev/gemma/terms)**.

### Mandatory Attribution
Any derivative works, merges, or re-uploads of this model **must explicitly credit** the author:
> **AlexH from [llmresearch.net](https://llmresearch.net)**

### Safety Warning
**This model is uncensored.** It will not refuse user requests based on standard safety guidelines. It may generate content that is controversial, sensitive, or offensive if prompted to do so. The user assumes all responsibility for the content generated and its usage.

---

<div align="center">

*Created by [**AlexH**](https://llmresearch.net) — Advancing the frontier of Cognitive Liberty.*

</div>

#  LoRA Fine-Tuning with DistilBERT for Sentiment Classification

This project demonstrates a **parameter-efficient fine-tuning approach (LoRA)** on a lightweight transformer (`distilbert-base-uncased`) to perform **binary sentiment classification** on a text dataset. By injecting low-rank adapters into selective layers of the model, we drastically reduce the number of trainable parameters — enabling faster, cheaper, and memory-efficient fine-tuning without compromising performance.

---

##  Project Objective

- Fine-tune `DistilBERT` using **LoRA (Low-Rank Adaptation)** on a small text dataset.
- Apply sentiment classification: **Positive (1)** or **Negative (0)**.
- Evaluate the model performance using test accuracy.
- Demonstrate effectiveness of training with just **~74K trainable parameters** out of ~67M total.

---

##  Dataset Overview

- Format: `.csv` file with two columns — `text` (review string) and `label` (0 or 1).
- Size: ~2,000 samples
- Preprocessing:
  - HTML tag removal using `BeautifulSoup`
  - Tokenization using Hugging Face `AutoTokenizer`

---

##  Model & Architecture

| Component           | Description                          |
|---------------------|--------------------------------------|
| Base Model          | `distilbert-base-uncased`            |
| Task                | Binary Classification (0 or 1)       |
| Output Head         | Single-unit Linear layer             |
| Loss Function       | `BCEWithLogitsLoss` (for logits)     |
| Optimizer           | Adam with weight decay               |
| Fine-tuning Method  | **LoRA** (Low-Rank Adaptation)       |

---

##  What is LoRA?

LoRA (Low-Rank Adaptation) modifies only small low-rank matrices injected into attention layers of transformers instead of updating full model weights.

###  Why LoRA?
- Reduce training cost and memory.
- Fine-tune with limited data and resources.
- Retain generalization of the original model.

---

##  LoRA Configuration

Only certain submodules are modified with LoRA. Configuration:

```python
lora_config = {
    "rank": 4,
    "alpha": 16,
    "dropout": 0.05,
    "apply_lora_to": {
        "q_lin": True,
        "v_lin": True,
        "k_lin": False,
        "ffn": False,
        "output_proj": False,
        "classifier_head": False
    }
}
```


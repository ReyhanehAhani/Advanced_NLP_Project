
# Robust Cross-Lingual Fact-Checking

This project investigates **multilingual fact-checking** using XLM-RoBERTa, with a focus on robustness to adversarial noise, parameter-efficient fine-tuning using **LoRA**, and evidence integration using **retrieval and summarization** pipelines. We address both **claim-only** settings and **evidence-rich** tasks across multiple languages.

## 🚀 Overview

1. **Cross-lingual Fact Verification:**  
   - Fine-tuned XLM-RoBERTa-base and large on the multilingual **X-Fact** dataset.
   - Evaluated robustness under synthetic character-level perturbations (10%, 30%, 60%).
   - Applied **adversarial fine-tuning** using noisy samples.
   - Conducted **gradient-based sensitivity analysis** across languages and labels.

2. **Monolingual Evidence-Based Verification (SciFact):**  
   - Developed a multi-stage **retrieval + summarization pipeline** using FLAN-T5, Wikipedia API, and BART.
   - Fine-tuned XLM-RoBERTa on SciFact and tested under four settings:
     - With and without fine-tuning
     - With and without evidence retrieval

3. **Baselines:**  
   - Compared performance against **LOKI**, an open-source multilingual fact-checking tool.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `finetune_lora.py` | Fine-tunes XLM-RoBERTa on X-Fact using LoRA |
| `evaluate_noise.py` | Tests model under different noise levels |
| `adversarial_training.py` | Performs adversarial fine-tuning |
| `gradient_analysis.py` | Computes gradient norms for sensitivity analysis |
| `sciFact_finetune.py` | Fine-tunes and evaluates on SciFact dataset |
| `retrieval_pipeline.py` | Retrieval + summarization pipeline using FLAN-T5 + BART |
| `utils.py` | Utilities for loading datasets, metrics, and preprocessing |
| `confusion_matrix_plots.ipynb` | Plots for visualizing evaluation results |
| `README.md` | Project overview and instructions (you are here) |

---

## 📊 Datasets

- **X-Fact (multilingual):**  
  - 9 languages: Farsi, Turkish, German, Spanish, Dutch, French, Polish, Georgian, Sinhala  
  - Labels: True, Mostly True, Mostly False, False (after remapping)  
  - Source: [X-Fact on HuggingFace](https://huggingface.co/datasets/utahnlp/x-fact)

- **SciFact (monolingual English, scientific):**  
  - Used for testing retrieval + fine-tuning pipeline  
  - Source: [SciFact on Kaggle](https://www.kaggle.com/datasets/thedevastator/unlock-insight-into-scientific-claims-with-scifa)

---

## 🔍 Approach Summary

### Multilingual Fine-Tuning (X-Fact)
- Model: `XLM-RoBERTa-{base,large}` + LoRA (r=128)
- Noise: Evaluated 10% / 30% / 60% char-level perturbations
- Metrics: F1 (weighted), Accuracy, Recall
- Robustness: Enhanced via adversarial training
- Hardware: 2× T4 GPUs (Kaggle Accelerators)

### Gradient Sensitivity
- Computed gradient norms of input embeddings
- Revealed which languages or label classes showed vulnerability

### Retrieval + Summarization (SciFact)
- Query reformulation: FLAN-T5 (zero-shot)
- Document retrieval: Wikipedia API
- Paragraph ranking: SentenceTransformer (multi-qa-mpnet-base-dot-v1)
- Summarization: facebook/bart-large-cnn (two-stage)

---

## 📈 Results

### Fine-Tuning vs. Unfined Models (X-Fact)
| Model | Accuracy (%) | F1 (%) | Recall (%) | Fine-tuned |
|-------|--------------|--------|-------------|-------------|
| XLM-R base | 31 | 15 | 31 | ❌ |
| XLM-R base | 39 | 40 | 39 | ✅ |
| XLM-R large | 11 | 2 | 11 | ❌ |
| XLM-R large | 44 | 45 | 44 | ✅ |

### Robustness to Perturbation
- F1 score and accuracy dropped significantly for languages like French and Dutch under heavy perturbations
- Stability observed in languages like Persian and Polish after adversarial fine-tuning

### SciFact + Retrieval
| Setting | Accuracy (%) | F1 (%) |
|---------|--------------|--------|
| Fine-tuned (no retrieval) | 60.05 | 58.74 |
| Fine-tuned (with retrieval) | 67.02 | 63.04 |
| Unfine-tuned (no retrieval) | 63.91 | 38.99 |
| Unfine-tuned (with retrieval) | 34.04 | 25.40 |

---

## 📦 Dependencies

- `transformers`, `datasets`, `peft`, `sentence-transformers`, `scikit-learn`, `torch`, `numpy`, `pandas`

Install with:

```bash
pip install -r requirements.txt
```

---

## 👩‍💻 Contributors

- **Reyhaneh Ahani** — Multilingual modeling, adversarial training, gradient analysis, SciFact experiments  
- **Pegah Aryadoost** — LOKI setup, retrieval integration, Wikipedia data prep  
- **Both** — Report writing, slides, and evaluation coordination

---

## 📜 Citation

If you use this project, please cite our report:

> R. Ahani, P. Aryadoost. "Cross-Lingual Fact-Checking". Simon Fraser University, 2025.

---

## 📂 Acknowledgments

We gratefully acknowledge:
- UtahNLP for X-Fact dataset
- AllenAI for SciFact
- HuggingFace & Kaggle for compute and dataset access

---

## License

MIT License © 2025 Reyhaneh Ahani & Pegah Aryadoost

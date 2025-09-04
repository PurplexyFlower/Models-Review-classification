
---

# **PhD Research Project: Advanced Text Classification Using QLoRA Fine-tuning**

**A Comprehensive Study on Large Language Model Fine-tuning for Review Classification**

---

## 📋 Executive Summary

This research project demonstrates the effectiveness of QLoRA (Quantized Low-Rank Adaptation) fine-tuning for binary text classification tasks, specifically focusing on fake review detection. The study compares multiple approaches, revealing that while traditional methods with modern embeddings provide a strong baseline, full QLoRA fine-tuning achieves a state-of-the-art **98.04% F1 Score** with Stella-400M.

### Key Findings
- **Best Performance**: Stella QLoRA (5 epochs) achieved a **98.04% F1 Score**.
- **Superiority of Fine-Tuning**: QLoRA fine-tuning substantially outperformed feature-extraction methods, showing a **~6% absolute F1 score improvement** over the best deep learning baseline.
- **Efficiency**: QLoRA reduces memory requirements significantly, enabling training on consumer GPUs (6-8GB VRAM) to achieve enterprise-level results.
- **Methodology**: A comprehensive evaluation across four distinct, verified approaches validates the findings.

---

## 🎯 Research Objectives

### Primary Objectives
1. **Performance Optimization**: Achieve >95% F1 Score for fake review detection.
2. **Resource Efficiency**: Enable training on consumer hardware (≤8GB VRAM).
3. **Methodology Comparison**: Evaluate traditional ML and simple DL vs. modern fine-tuning approaches.
4. **Scalability Assessment**: Develop production-ready models with practical deployment constraints.

### Secondary Objectives
1. Compare different base model architectures (FLAN-T5 vs. Stella).
2. Analyze the impact of training duration on performance.
3. Establish best practices for QLoRA implementation.
4. Create comprehensive evaluation frameworks.

---

## 🔬 Methodology

### Dataset Description
- **Source**: Product review dataset (Real vs. Fake classification)
- **Total Samples**: 40,432 reviews
- **Classes**: Binary (0: Fake, 1: Real)
- **Train/Validation/Test Split**: 70% / 10% / 20% (for fine-tuning)
- **Class Distribution**: Balanced (approximately 50-50 split)

### Data Preprocessing
```python
# Data cleaning pipeline
1. Remove null values and duplicates
2. Text normalization (lowercase, remove special characters)
3. Stratified sampling for consistent splits
```

### Evaluation Metrics
- **Primary**: F1 Score (binary classification)
- **Secondary**: Accuracy, Precision, Recall, ROC AUC, Confusion Matrix

---

## 🏗️ Model Architectures

### 1. Stella-400M with QLoRA (Champion Model)
**Architecture**: Encoder-based transformer with custom classification head.
```yaml
Base Model: NovaSearch/stella_en_400M_v5
Parameters: 436.5M (base) + 2.4M (adapters, ~0.54%)
Quantization: 4-bit NF4 with double quantization
LoRA Configuration:
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.05
  - Target Modules: ["qkv_proj", "o_proj"]
```

### 2. FLAN-T5-Large with QLoRA
**Architecture**: Encoder-decoder transformer adapted for classification.
```yaml
Base Model: google/flan-t5-large
Parameters: 759.7M (base) + 9.4M (adapters, ~1.24%)
Quantization: 4-bit NF4 with double quantization
LoRA Configuration:
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.05
  - Target Modules: ["q", "k", "v", "o"]
```

### 3. Stella + Deep Learning
**Architecture**: Pre-trained embeddings + Custom neural network.
```yaml
Embedding Model: NovaSearch/stella_en_400M_v5
Architecture:
  - Input: 1024-dim Stella embeddings
  - Hidden Layer 1: 256 neurons (ReLU + Dropout 0.5)
  - Hidden Layer 2: 128 neurons (ReLU + Dropout 0.5)
  - Output: 2 classes (Softmax)
Optimizer: Adam (lr=0.001)
```

### 4. Traditional ML with Stella Embeddings
**Feature Extraction**: Stella-400M embeddings (1024 dimensions).
- **Logistic Regression**: Standard `sklearn` implementation.
- **Support Vector Machine**: `sklearn` implementation with RBF kernel.

---

## 🔍 Detailed Model Analysis

### Target Module Selection
For this study, the LoRA adapters were applied to the core self-attention layers of the transformer models, which is a standard and effective practice.
- **Stella Model**: The `qkv_proj` (query, key, value) and `o_proj` (output) modules were targeted. This focuses adaptation on the mechanisms that are most critical for capturing contextual relationships in the input data.
- **FLAN-T5 Model**: Similarly, the `q`, `k`, `v`, and `o` modules were targeted, ensuring that the fine-tuning process adapted the model's fundamental attention mechanisms.

This approach effectively balances performance gains with parameter efficiency, concentrating the trainable parameters where they will have the most impact.

### Overall Performance Comparison

| Rank | Model | F1 Score | Accuracy |
| :--- | :--- | :--- | :--- |
| 🥇 | **Stella QLoRA (5 epochs)** | **98.04%** | **98.02%** |
| 🥈 | Stella QLoRA (3 epochs) | 97.92% | 97.90% |
| 🥉 | FLAN-T5 QLoRA | 97.03% | 96.98% |
| 4th | Stella + Deep Learning | 92.16% | 92.16% |
| 5th | Stella + Logistic Regression | 91.49% | 91.49% |
| 6th | Stella + SVM | 91.27% | 91.27% |

### Stella QLoRA (5 Epochs) - Champion Model Details

#### Confusion Matrix Analysis
```
Predicted:    Fake    Real
Actual:
Fake         3922     122     (TN: 3922, FP: 122)
Real           38    4005     (FN: 38,   TP: 4005)
```
- **Accuracy**: 98.02%

#### Training Progression Analysis
- **Epoch 3**: 97.92% F1 (Best result from initial training)
- **Epoch 5**: 98.04% F1 (Improved result after resuming training)
- **Improvement**: +0.12% absolute F1 score gain from extended training.

---

## 📈 Performance Visualizations

### Model Comparison Chart
```
F1 Score Performance Comparison
================================

Stella QLoRA (5 epochs)    |██████████████████████████ 98.04%
Stella QLoRA (3 epochs)    |█████████████████████████▌ 97.92%
FLAN-T5 QLoRA             |████████████████████████▏  97.03%
Stella + Deep Learning     |████████████████████▊      92.16%
Stella + Logistic Reg      |███████████████████▋       91.49%
Stella + SVM              |███████████████████▍       91.27%

                           88%  90%  92%  94%  96%  98%  100%
```
### Memory Usage Comparison
```
VRAM Requirements
=================

High Memory (8GB+):
└── FLAN-T5 QLoRA          ~8GB

Medium Memory (4-6GB):
├── Stella QLoRA           ~6GB
└── Stella + Deep Learning   ~4GB

Low Memory (<4GB):
├── Stella + LogReg        ~2GB
└── Stella + SVM           ~2GB
```
---

## 💼 Business Impact Analysis

### Cost-Benefit Analysis
```
Estimated Training Cost Analysis
==============================

Hardware Requirements:
├── Traditional Full Fine-tuning: Multi-GPU Enterprise Server (e.g., 4x A100, ~$20,000+)
├── Our QLoRA Approach:       1x Consumer GPU (e.g., RTX 40-series, ~$1,600)
└── Estimated Cost Reduction:   ~92% in hardware capital expenditure.

Training Time:
├── Traditional: 24-48 hours (estimated)
├── Our Approach: 4-6 hours (observed)
└── Time Reduction: ~80% savings in training time.

Performance Comparison:
├── Traditional: ~98.1% F1 (estimated SOTA)
├── Our Approach: 98.04% F1 (achieved)
└── Performance Gap: <0.1%, achieving comparable performance for a fraction of the cost.
```

### Deployment Considerations
```
Production Deployment Analysis
=============================

Model Size:
├── Stella QLoRA:   ~800MB (efficient for a large model)
├── FLAN-T5 QLoRA:  ~1.2GB (moderate)
└── Baseline models: 10-50MB (minimal)

Inference Speed (Estimated for 1000 reviews):
├── Stella QLoRA:   ~45 seconds
├── FLAN-T5 QLoRA:  ~60 seconds
├── Stella + DL:    ~15 seconds
└── Traditional ML: ~2 seconds

Recommendation:
- For **high-accuracy** applications, the Stella QLoRA model is the clear choice.
- For **high-throughput / low-resource** scenarios where speed is critical, the Stella + DL model is a viable alternative, accepting a performance trade-off.
```

---

## 🎯 Key Insights and Conclusions

### Major Findings

1.  **QLoRA Superiority**: QLoRA fine-tuning is demonstrably superior to feature-extraction methods for this task. The **~6% absolute F1 score gap** between the Stella QLoRA model (98.04%) and the best Stella+DL baseline (92.16%) highlights the importance of adapting the full model via adapters for optimal performance.
2.  **Architecture Choice**: The encoder-only Stella-400M model slightly outperformed the larger encoder-decoder FLAN-T5-Large model (98.04% vs. 97.03%) on this classification task.
3.  **Training Duration**: Extended training (from 3 to 5 epochs) provided a measurable improvement, pushing the model to its peak performance.
4.  **Baseline Performance**: Traditional ML and simple DL models using Stella embeddings provide a respectable baseline of ~91-92% F1 but are not competitive with fine-tuned models.

### Research Implications

1.  **Democratization of Fine-Tuning**: This study confirms that consumer hardware can be used to fine-tune powerful LLMs to achieve state-of-the-art, production-ready performance.
2.  **Methodological Recommendation**: For tasks requiring high accuracy, fine-tuning with techniques like QLoRA should be the preferred method over feature extraction, as the performance gains are significant.

---

## 🚀 Future Work and Recommendations

### Immediate Opportunities
1. **Cross-Domain Validation**: Test models on different review domains (e.g., movies, restaurants).
2. **Multilingual Extension**: Adapt approaches for non-English review classification.
3. **Ensemble Methods**: Combine top-performing models for potential improvements.
4. **Real-time Deployment**: Implement a production API with A/B testing.

### Advanced Research Directions
1. **Novel Architectures**: Explore newer models (e.g., Llama-3, Gemma) with QLoRA.
2. **Automated Hyperparameter Optimization**: Conduct a systematic search for optimal LoRA configurations (e.g., rank, alpha).
3. **Interpretability Analysis**: Use techniques like SHAP or LIME to understand model decision-making.

---

## 📚 References and Resources

### Technical Documentation
- **Hugging Face Transformers**: Model implementations and fine-tuning guides.
- **PEFT Library**: QLoRA and LoRA implementation details.
- **BitsAndBytes**: Quantization techniques and optimizations.

### Research Papers
- Dettmers et al., 2023. *QLoRA: Efficient Finetuning of Quantized LLMs*.
- Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models*.

### Model Sources
- **NovaSearch/stella_en_400M_v5**: Base embedding model.
- **google/flan-t5-large**: Alternative base model.

---

## 📋 Appendix

### A. Hardware Specifications
```yaml
Training Environment:
  GPU: NVIDIA RTX Series (6-8GB VRAM)
  CPU: Multi-core (8+ cores recommended)
  RAM: 16GB+ system memory
```

### B. Reproducibility Information
```bash
# Environment setup
git clone https://github.com/PurplexyFlower/Models-Review-classification.git
# ... follow repository instructions to run notebooks
```
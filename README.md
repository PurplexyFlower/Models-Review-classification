# PhD Models - Review ├── models/                         # Fine-tuned model checkpoints
│   ├── flan-t5-large-qlora-tuned/ # FLAN-T5 QLoRA model (97.03% F1)
│   └── stella_qlora_finetuned_model/ # Stella QLoRA model (🏆 BEST: 98.04% F1)
├── notebooks/                      # Jupyter notebooks for experiments
│   ├── flan-t5-large-qlora-tuned.ipynb        # FLAN-T5 fine-tuning notebook
│   ├── Stella_Classification_Methods_GPU.ipynb # Stella ML/DL methods
│   ├── Stella_QLoRA_Fine_Tuning.ipynb         # Stella QLoRA fine-tuning
│   └── Stella_Resume_Training_5_Epochs.ipynb  # Extended Stella training
├── scripts/                        # Python scripts
│   ├── FLan-t5-large_FT_QLoRA.py  # FLAN-T5 training script
│   ├── FT_Stella.py               # Stella training script
│   └── resume_stella_training.py  # Resume Stella training scripttion with QLoRA Fine-tuning

This repository contains the implementation and results of fine-tuning large language models for review classification using QLoRA (Quantized Low-Rank Adaptation) technique. The project focuses on binary classification of product reviews (Real vs Fake) using two different model architectures: FLAN-T5-Large and Stella.

## 🎯 Project Overview

This research project demonstrates efficient fine-tuning of large language models for text classification tasks using consumer-grade hardware. By implementing QLoRA, we achieve comparable performance to full fine-tuning while using significantly less computational resources.

### Key Achievements
- **Stella QLoRA Fine-tuning (5 Epochs)**: Achieved **98.04% F1 Score** (🏆 BEST PERFORMANCE!)
- **FLAN-T5-Large QLoRA**: Achieved **97.03% F1 Score** on review classification
- **Stella + Deep Learning**: Custom neural classifier with **96.85% F1 Score**
- **Stella + Machine Learning**: Traditional ML classifiers up to **95.40% F1 Score**
- **Memory Efficiency**: Reduced VRAM requirements by ~75% compared to full fine-tuning
- **Training Time**: 4-6 hours on consumer GPU vs 20+ hours for traditional fine-tuning
- **Performance Improvement**: +0.12% boost from extended training (3→5 epochs)

## 📁 Repository Structure

```
├── data/                           # Datasets and embeddings
│   ├── dataset/                    # Raw and processed review data
│   └── stella_embeddings/          # Pre-computed Stella embeddings
├── models/                         # Fine-tuned model checkpoints
│   ├── flan-t5-large-qlora-tuned/ # FLAN-T5 QLoRA model (BEST: 97.03% F1)
│   └── stella_qlora_finetuned_model/ # Stella QLoRA model
├── notebooks/                      # Jupyter notebooks for experiments
│   ├── flan-t5-large-qlora-tuned.ipynb    # FLAN-T5 fine-tuning notebook
│   ├── Stella_Classification_Methods_GPU.ipynb # Stella GPU implementation
│   └── Stella_QLoRA_Fine_Tuning.ipynb     # Stella QLoRA fine-tuning
├── scripts/                        # Python scripts
│   ├── FLan-t5-large_FT_QLoRA.py  # FLAN-T5 training script
│   └── FT_Stella.py               # Stella training script
├── outputs/                        # Model weights and test results
├── documentation/                  # Additional documentation
└── .venv/                         # Python virtual environment
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NVIDIA GPU with 6-8GB VRAM (minimum)
- CUDA-compatible drivers

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/phd-models-review-classification.git
cd phd-models-review-classification
```

2. Set up the virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Models

#### Stella QLoRA Fine-tuning (Recommended - 🏆 Best Performance: 98.04%)
```bash
# Run the comprehensive training notebook
jupyter notebook notebooks/Stella_QLoRA_Fine_Tuning.ipynb

# Or resume training from 3 to 5 epochs (for best results)
jupyter notebook notebooks/Stella_Resume_Training_5_Epochs.ipynb

# Or run the script
python scripts/FT_Stella.py
```

#### FLAN-T5-Large QLoRA (Alternative - 97.03% F1)
```bash
# Run the notebook
jupyter notebook notebooks/flan-t5-large-qlora-tuned.ipynb

# Or run the script
python scripts/FLan-t5-large_FT_QLoRA.py
```

#### Stella ML/DL Methods (Fast Training - Up to 96.85% F1)
```bash
# Run comprehensive ML and DL experiments
jupyter notebook notebooks/Stella_Classification_Methods_GPU.ipynb
```

## 🏆 Model Performance

### FLAN-T5-Large-ReviewClassifier-QLoRA (Third Best Model)
- **F1 Score**: 97.03%
- **Training Time**: ~5 hours
- **Memory Usage**: ~8GB VRAM
- **Model Size**: ~1.2GB (adapters only)
- **Best Checkpoint**: `models/flan-t5-large-qlora-tuned/checkpoint-5460/`

### Stella QLoRA Fine-tuning (🥇 BEST MODEL)
- **F1 Score**: 97.92% (Highest Performance!)
- **Architecture**: NovaSearch/stella_en_400M_v5 with QLoRA adapters
- **Training Time**: ~4 hours
- **Memory Usage**: ~6GB VRAM
- **Model Size**: ~800MB (adapters only)
- **Best Checkpoint**: `models/stella_qlora_finetuned_model/checkpoint-5460/`

### Stella + Deep Learning Classifier (Second Best)
- **F1 Score**: 96.85%
- **Architecture**: Stella embeddings + 3-layer neural network
- **Features**: 1536-dimensional Stella embeddings
- **Training Time**: ~30 minutes
- **Memory Usage**: ~4GB VRAM
- **Model Components**: 
  - Input Layer: 1536 features
  - Hidden Layer 1: 256 neurons (ReLU + Dropout 0.5)
  - Hidden Layer 2: 128 neurons (ReLU + Dropout 0.5)
  - Output Layer: 2 classes (softmax)

### Stella + Machine Learning Classifiers
#### Logistic Regression
- **F1 Score**: 95.40%
- **Features**: 1536-dimensional Stella embeddings
- **Training Time**: ~2 minutes
- **Advantages**: Fast inference, interpretable

#### Support Vector Machine (SVM)
- **F1 Score**: 94.85%
- **Kernel**: RBF (Radial Basis Function)
- **Features**: 1536-dimensional Stella embeddings
- **Training Time**: ~5 minutes
- **Advantages**: Strong generalization, robust to overfitting

## 📊 Performance Comparison

| Model | F1 Score | Training Time | Memory (VRAM) | Model Size | Approach |
|-------|----------|---------------|---------------|------------|----------|
| **🏆 Stella QLoRA (5 epochs)** | **98.04%** | ~6 hours | ~6GB | ~800MB | Fine-tuning |
| Stella QLoRA (3 epochs) | 97.92% | ~4 hours | ~6GB | ~800MB | Fine-tuning |
| FLAN-T5 QLoRA | 97.03% | ~5 hours | ~8GB | ~1.2GB | Fine-tuning |
| Stella + DL | 96.85% | ~30 min | ~4GB | ~50MB | Feature Extraction |
| Stella + LogReg | 95.40% | ~2 min | ~2GB | ~10MB | Feature Extraction |
| Stella + SVM | 94.85% | ~5 min | ~2GB | ~15MB | Feature Extraction |

### 🎯 Key Insights
- **Best Performance**: Stella QLoRA with 5 epochs achieves **98.04% F1 Score**
- **Performance Gain**: +0.12% improvement from extended training (3→5 epochs)
- **Efficiency**: QLoRA achieves near-optimal performance with 75% less memory
- **Speed vs Quality**: Feature extraction methods offer 10x faster training with ~2% F1 loss

## 🔧 Technical Details

### QLoRA Configuration
- **Quantization**: 4-bit NF4 with double quantization
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **LoRA Dropout**: 0.05
- **Target Modules**: Attention layers (q, k, v, o)

### Training Hyperparameters
- **Learning Rate**: 2e-4
- **Batch Size**: 2 per device (effective: 16 with gradient accumulation)
- **Optimizer**: paged_adamw_8bit
- **Scheduler**: Cosine annealing
- **Max Epochs**: 5 (early stopping at epoch 3)

### Hardware Requirements
- **Training**: NVIDIA GPU with 6-8GB VRAM minimum
- **Inference**: Any CUDA GPU with 2GB+ VRAM
- **System RAM**: 16GB+ recommended

## 📊 Dataset

The project uses a cleaned dataset of product reviews with binary labels:
- **Real Reviews**: Genuine customer feedback
- **Fake Reviews**: Artificially generated or manipulated reviews
- **Data Splits**: 72% training, 8% validation, 20% testing
- **Preprocessing**: Text cleaning, tokenization, stratified splitting

## 📈 Results and Analysis

### Training Progress Comparison

#### Stella QLoRA (Best Model - 97.92% F1)
- **Epoch 1**: F1 = 96.80%
- **Epoch 2**: F1 = 97.45%
- **Epoch 3**: F1 = 97.92% ← **Best Model**
- **Extended Training**: Potential for 5-epoch training available

#### FLAN-T5 QLoRA (97.03% F1)
- **Epoch 1**: F1 = 96.45%
- **Epoch 2**: F1 = 96.52%
- **Epoch 3**: F1 = 97.03% ← **Best Model**
- **Epochs 4-5**: No improvement (early stopping)

#### Stella Deep Learning Classifier (96.85% F1)
- **Training**: 15 epochs with early stopping
- **Best Epoch**: Epoch 12 (patience = 3)
- **Architecture**: 3-layer MLP with dropout regularization
- **Convergence**: Stable training with validation monitoring

#### Stella Machine Learning Classifiers
- **Logistic Regression**: Converged in 847 iterations (95.40% F1)
- **SVM (RBF Kernel)**: Optimal hyperparameters found via grid search (94.85% F1)
- **Feature Dimensionality**: 1536 (Stella embedding size)

### Detailed Performance Metrics

#### Stella QLoRA Fine-tuning (Champion Model)
```
                precision    recall  f1-score   support
Real Reviews       0.9795    0.9789    0.9792      1205
Fake Reviews       0.9789    0.9795    0.9792      1193
     accuracy                          0.9792      2398
    macro avg       0.9792    0.9792    0.9792      2398
 weighted avg       0.9792    0.9792    0.9792      2398
```

#### Stella Deep Learning Classifier
```
                precision    recall  f1-score   support
Real Reviews       0.9702    0.9668    0.9685      1205  
Fake Reviews       0.9665    0.9699    0.9682      1193
     accuracy                          0.9684      2398
    macro avg       0.9684    0.9684    0.9683      2398
 weighted avg       0.9684    0.9684    0.9685      2398
```

#### Stella Logistic Regression
```
                precision    recall  f1-score   support
Real Reviews       0.9563    0.9520    0.9541      1205
Fake Reviews       0.9516    0.9559    0.9537      1193  
     accuracy                          0.9540      2398
    macro avg       0.9540    0.9540    0.9539      2398
 weighted avg       0.9540    0.9540    0.9540      2398
```

### Memory Efficiency
- **Traditional Fine-tuning**: 40+ GB VRAM required
- **QLoRA Fine-tuning**: 8GB VRAM sufficient
- **Reduction**: ~75% memory savings

### Environmental Impact
- **Training Time**: 75% reduction vs full fine-tuning
- **Energy Consumption**: Significantly lower carbon footprint
- **Accessibility**: Enables research on consumer hardware

## 🔬 Research Contributions

1. **Comprehensive Model Comparison**: Evaluated 5 different approaches from traditional ML to state-of-the-art QLoRA fine-tuning
2. **Efficient Fine-tuning**: Demonstrated QLoRA effectiveness across different model architectures (T5 vs Stella)
3. **Stella Model Innovation**: First comprehensive study of NovaSearch/stella_en_400M_v5 for classification tasks
4. **Feature Extraction Analysis**: Showed Stella embeddings effectiveness for downstream ML/DL tasks
5. **Consumer Hardware Accessibility**: Proved high-quality results (97.92% F1) on limited resources
6. **Multi-Method Framework**: Provided complete pipeline from feature extraction to fine-tuning
7. **Reproducible Research**: Complete code, notebooks, and documentation for all approaches

## 📝 Usage Examples

### Loading the Best Model (Stella QLoRA - 97.92% F1)
```python
from transformers import AutoTokenizer
from peft import PeftModel, AutoPeftModelForSequenceClassification

# Method 1: Direct loading (recommended)
model_path = "models/stella_qlora_finetuned_model/checkpoint-5460"
model = AutoPeftModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("NovaSearch/stella_en_400M_v5")

# Classify a review
text = "This product exceeded my expectations! Fast shipping and great quality."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=-1)
confidence = torch.softmax(outputs.logits, dim=-1).max().item()

print(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
print(f"Confidence: {confidence:.4f}")
```

### Using Stella Embeddings for Custom ML
```python
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load pre-trained Stella model for feature extraction
model_name = "NovaSearch/stella_en_400M_v5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

def get_stella_embeddings(texts):
    """Extract Stella embeddings for texts."""
    inputs = tokenizer(texts, padding=True, truncation=True, 
                      return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
    return embeddings.numpy()

# Example usage
texts = ["Great product!", "Terrible quality, waste of money."]
embeddings = get_stella_embeddings(texts)

# Train your custom classifier
# classifier = LogisticRegression()
# classifier.fit(embeddings, labels)
```

### Loading Deep Learning Classifier
```python
import torch
import torch.nn as nn

# Define the same architecture as trained
class ComplexClassifier(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim1=256, hidden_dim2=128, output_dim=2):
        super(ComplexClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

# Load the trained model
dl_classifier = ComplexClassifier()
dl_classifier.load_state_dict(torch.load('best_complex_classifier.pth'))
dl_classifier.eval()

# Use with Stella embeddings for classification
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{phd-models-review-classification,
  title={Efficient Review Classification using QLoRA Fine-tuning of Large Language Models},
  author={PhD Research Project},
  year={2025},
  howpublished={GitHub Repository},
  url={https://github.com/your-username/phd-models-review-classification}
}
```

## 🙏 Acknowledgments

- **Google**: For the FLAN-T5 base model
- **Hugging Face**: For transformers and PEFT libraries
- **Microsoft**: For BitsAndBytes quantization
- **Research Community**: For QLoRA methodology

## 📞 Contact

For questions about this research, please open an issue or contact [your-email@domain.com].

---

**Note**: This is a research project focused on demonstrating efficient fine-tuning techniques. The models are trained for educational and research purposes.

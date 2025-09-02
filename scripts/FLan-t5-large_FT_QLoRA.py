# ==============================================================================
# BLOCK 1: IMPORT LIBRARIES
# ------------------------------------------------------------------------------
# All necessary libraries are imported at the beginning for clarity.
# ==============================================================================
import os
import sys
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification, # Using the standard class for T5
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training


# ==============================================================================
# BLOCK 2: CONFIGURATION
# ------------------------------------------------------------------------------
# All important parameters are defined here, making it easy to experiment.
# ==============================================================================
# --- Model and Data Configuration ---
MODEL_NAME = "google/flan-t5-large"
DATASET_PATH = "dataset/cleaned_dataset.csv"
OUTPUT_DIR = "./flan-t5-large-qlora-tuned"

# --- QLoRA (PEFT) Hyperparameters (tuned for a 6GB GPU) ---
LORA_R = 16          # LoRA rank. 16 is a good balance of performance and memory.
LORA_ALPHA = 32      # LoRA alpha, typically 2 * LORA_R.
LORA_DROPOUT = 0.05  # Dropout for the LoRA layers to prevent overfitting.
# Target modules for T5 models are typically 'q', 'k', 'v', 'o'.
LORA_TARGET_MODULES = ["q", "k", "v", "o"]

# --- Training Hyperparameters ---
BATCH_SIZE = 2      # Per-device batch size. Must be small for 6GB VRAM.
GRADIENT_ACCUMULATION_STEPS = 8 # Simulates a larger batch size. Effective batch size = 2 * 8 = 16.
LEARNING_RATE = 2e-4  # Recommended learning rate for QLoRA.
NUM_EPOCHS = 5   # Number of times to train on the entire dataset.
MAX_SEQ_LENGTH = 512 # The documented max sequence length for Flan-T5.


# ==============================================================================
# BLOCK 3: HELPER FUNCTIONS
# ------------------------------------------------------------------------------
# These functions handle repetitive tasks like setting up the environment,
# loading data, and calculating metrics.
# ==============================================================================
def setup_environment():
    """Checks for GPU availability and returns the device and bfloat16 support status."""
    if not torch.cuda.is_available():
        print("❌ No GPU detected. QLoRA requires a GPU. Exiting.")
        sys.exit(1)
    
    device = torch.device("cuda")
    has_bf16_support = torch.cuda.is_bf16_supported()
    
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"✅ BFloat16 Support: {'Yes' if has_bf16_support else 'No (will use Float16)'}")
    
    torch.cuda.empty_cache()
    return device, has_bf16_support

def load_and_prepare_data(dataset_path):
    """Loads, cleans, and splits the data into training, validation, and test sets."""
    print(f"\n📂 Loading dataset from '{dataset_path}'...")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"❌ Dataset not found! Please check the path: {dataset_path}")
        sys.exit(1)

    df_clean = df[['text', 'label']].copy().dropna()
    
    train_val_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean['label'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42, stratify=train_val_df['label'])

    print("📈 Data splits created:")
    print(f"   - Training:   {len(train_df)} samples")
    print(f"   - Validation: {len(val_df)} samples")
    print(f"   - Test:       {len(test_df)} samples")
    
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df), Dataset.from_pandas(test_df)

def compute_metrics(eval_pred):
    """Calculates accuracy, F1, precision, and recall for evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


# ==============================================================================
# BLOCK 4: MAIN FINE-TUNING ORCHESTRATION
# ==============================================================================
def main():
    """Main function to run the entire fine-tuning and evaluation pipeline."""
    
    # --- Step 1: Initial Setup ---
    device, has_bf16_support = setup_environment()
    
    # --- Step 2: Configure QLoRA ---
    print("\n⚙️  Configuring QLoRA for 4-bit quantization...")
    compute_dtype = torch.bfloat16 if has_bf16_support else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # --- Step 3: Load Model and Tokenizer ---
    print(f"🚀 Loading base model '{MODEL_NAME}' with 4-bit quantization...")
    # We use the standard AutoModelForSequenceClassification class here
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        quantization_config=quantization_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # --- Step 4: Apply LoRA Adapters ---
    print("✨ Applying LoRA adapters to the model...")
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        # T5 is a text-to-text model, so the task type is SEQ_2_SEQ_LM
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)
    
    print("\n🔥 LoRA Model Configuration:")
    model.print_trainable_parameters()

    # --- Step 5: Process Datasets ---
    train_dataset, val_dataset, test_dataset = load_and_prepare_data(DATASET_PATH)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)
    print("\n... Tokenizing all datasets ...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # --- Step 6: Configure and Run the Trainer ---
    print("📋 Setting up Hugging Face Trainer...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        fp16=not has_bf16_support,
        bf16=has_bf16_support,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=25,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )
    
    print("\n🚀 Starting QLoRA fine-tuning...")
    trainer.train()
    print("✅ Fine-tuning complete!")
    
    # --- Step 7: Final Evaluation and Reporting ---
    print("\n🧪 Evaluating final model on the unseen test set...")
    predictions = trainer.predict(tokenized_test)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    print("\n" + "="*50)
    print("📊 Final Classification Report:")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=["Real Review", "Fake Review"]))

    print("\n" + "="*50)
    print("📈 Confusion Matrix:")
    print("="*50)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title(f'Confusion Matrix - {MODEL_NAME} on Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# ==============================================================================
# BLOCK 5: SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    main()
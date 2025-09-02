#!/usr/bin/env python3
"""
Resume Training Script for Stella QLoRA Model
============================================

This script resumes training the Stella model from the last checkpoint (5460)
to complete a full 5-epoch training cycle.

Current state:
- Completed 3 epochs (best F1: 97.92% at step 5460)
- Need to train 2 more epochs to reach 5 total epochs

Key Features:
- Resumes from existing checkpoint
- Preserves training state and optimizer
- Continues with same hyperparameters
- Monitors for improvements beyond current best (97.92%)
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    prepare_model_for_kbit_training,
    PeftModel
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
print("🔄 RESUMING STELLA QLORA TRAINING")
print("=" * 50)

# Model and paths
MODEL_NAME = "NovaSearch/stella_en_400M_v5"
DATASET_PATH = "data/dataset/cleaned_dataset.csv"
CHECKPOINT_PATH = "models/stella_qlora_finetuned_model/checkpoint-5460"
OUTPUT_DIR = "models/stella_qlora_finetuned_model"

# Training Configuration (maintaining same parameters)
TOTAL_EPOCHS = 5  # Target total epochs
COMPLETED_EPOCHS = 3  # Already completed
REMAINING_EPOCHS = TOTAL_EPOCHS - COMPLETED_EPOCHS  # 2 more epochs

# Original hyperparameters (keep consistent)
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512

print(f"📋 Resume Training Configuration:")
print(f"   🎯 Model: {MODEL_NAME}")
print(f"   📁 Checkpoint: {CHECKPOINT_PATH}")
print(f"   🔄 Total epochs: {TOTAL_EPOCHS}")
print(f"   ✅ Completed: {COMPLETED_EPOCHS}")
print(f"   ⏳ Remaining: {REMAINING_EPOCHS}")
print(f"   🎯 Current best F1: 97.92%")

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================
def setup_environment():
    """Setup and validate the training environment."""
    if not torch.cuda.is_available():
        print("❌ No GPU detected. QLoRA requires a GPU for efficient training.")
        return None, None
    
    device = torch.device("cuda")
    has_bf16_support = torch.cuda.is_bf16_supported()
    
    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"\\n✅ GPU detected: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    print(f"🔥 BFloat16 Support: {'Yes' if has_bf16_support else 'No (using FP16)'}")
    
    # Clear cache
    torch.cuda.empty_cache()
    
    return device, has_bf16_support

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_and_prepare_data(dataset_path):
    """Load and prepare the dataset (same as original training)."""
    print(f"\\n📂 Loading dataset from '{dataset_path}'...")
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded successfully: {len(df)} samples")
    except FileNotFoundError:
        print(f"❌ Dataset not found! Please check the path: {dataset_path}")
        return None, None, None
    
    # Clean data
    df_clean = df[['text', 'label']].copy().dropna()
    print(f"📊 After cleaning: {len(df_clean)} samples")
    
    # Same splits as original training (using same random_state for consistency)
    train_val_df, test_df = train_test_split(
        df_clean, test_size=0.2, random_state=42, stratify=df_clean['label']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.1, random_state=42, stratify=train_val_df['label']
    )
    
    print("📈 Data splits (same as original):")
    print(f"   - Training:   {len(train_df)} samples")
    print(f"   - Validation: {len(val_df)} samples")
    print(f"   - Test:       {len(test_df)} samples")
    
    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, val_dataset, test_dataset

# ==============================================================================
# MODEL LOADING FROM CHECKPOINT
# ==============================================================================
def load_model_from_checkpoint(checkpoint_path, device, has_bf16_support):
    """Load the model from the specified checkpoint."""
    print(f"\\n🔄 Loading model from checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None, None
    
    # Setup quantization (same as original)
    compute_dtype = torch.bfloat16 if has_bf16_support else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model with quantization
    print("🤖 Loading base model with quantization...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=compute_dtype
    )
    
    # Prepare for k-bit training
    base_model = prepare_model_for_kbit_training(base_model)
    
    # Load PEFT model from checkpoint
    print("🔗 Loading PEFT adapters from checkpoint...")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    
    print("✅ Model loaded successfully from checkpoint!")
    model.print_trainable_parameters()
    
    return model, tokenizer

# ==============================================================================
# COMPREHENSIVE EVALUATION FUNCTIONS
# ==============================================================================
def plot_confusion_matrix(y_true, y_pred, model_name="Stella", save_path=None):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'{model_name} - Confusion Matrix (5 Epochs)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return cm

def plot_roc_curve(y_true, y_proba, model_name="Stella", save_path=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve (5 Epochs)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 ROC curve saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_proba, model_name="Stella", save_path=None):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve (5 Epochs)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 PR curve saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return avg_precision

def generate_comprehensive_report(y_true, y_pred, y_proba, model_name="Stella", epoch=5):
    """Generate comprehensive evaluation report."""
    print(f"\n🔍 COMPREHENSIVE EVALUATION REPORT - {model_name} ({epoch} Epochs)")
    print("=" * 80)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # Advanced metrics
    roc_auc = roc_auc_score(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity and sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("📊 OVERALL PERFORMANCE METRICS:")
    print(f"   • Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   • F1-Score (Binary):  {f1_avg:.4f} ({f1_avg*100:.2f}%)")
    print(f"   • Precision (Binary): {precision_avg:.4f} ({precision_avg*100:.2f}%)")
    print(f"   • Recall (Binary):    {recall_avg:.4f} ({recall_avg*100:.2f}%)")
    print(f"   • ROC AUC:            {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    print(f"   • Average Precision:  {avg_precision:.4f} ({avg_precision*100:.2f}%)")
    
    print("\n🎯 CLASS-WISE PERFORMANCE:")
    print(f"   📋 Class 0 (Negative):")
    print(f"      - Precision: {precision[0]:.4f} ({precision[0]*100:.2f}%)")
    print(f"      - Recall:    {recall[0]:.4f} ({recall[0]*100:.2f}%)")
    print(f"      - F1-Score:  {f1[0]:.4f} ({f1[0]*100:.2f}%)")
    print(f"      - Support:   {support[0]} samples")
    
    print(f"   📋 Class 1 (Positive):")
    print(f"      - Precision: {precision[1]:.4f} ({precision[1]*100:.2f}%)")
    print(f"      - Recall:    {recall[1]:.4f} ({recall[1]*100:.2f}%)")
    print(f"      - F1-Score:  {f1[1]:.4f} ({f1[1]*100:.2f}%)")
    print(f"      - Support:   {support[1]} samples")
    
    print("\n🔢 CONFUSION MATRIX ANALYSIS:")
    print(f"   📊 Confusion Matrix:")
    print(f"      [[{tn:4d}, {fp:4d}]  <- [TN, FP]")
    print(f"       [{fn:4d}, {tp:4d}]] <- [FN, TP]")
    
    print(f"   📈 Detailed Breakdown:")
    print(f"      • True Negatives:  {tn:4d} ({tn/(tn+fp+fn+tp)*100:.1f}%)")
    print(f"      • False Positives: {fp:4d} ({fp/(tn+fp+fn+tp)*100:.1f}%)")
    print(f"      • False Negatives: {fn:4d} ({fn/(tn+fp+fn+tp)*100:.1f}%)")
    print(f"      • True Positives:  {tp:4d} ({tp/(tn+fp+fn+tp)*100:.1f}%)")
    
    print(f"   🎯 Additional Metrics:")
    print(f"      • Specificity (TNR): {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"      • Sensitivity (TPR): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"      • False Positive Rate: {fp/(fp+tn):.4f} ({fp/(fp+tn)*100:.2f}%)")
    print(f"      • False Negative Rate: {fn/(fn+tp):.4f} ({fn/(fn+tp)*100:.2f}%)")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1_avg,
        'precision': precision_avg,
        'recall': recall_avg,
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'confusion_matrix': cm,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'class_wise_metrics': {
            'class_0': {'precision': precision[0], 'recall': recall[0], 'f1': f1[0], 'support': support[0]},
            'class_1': {'precision': precision[1], 'recall': recall[1], 'f1': f1[1], 'support': support[1]}
        }
    }

def save_results_to_file(results, model_name="Stella", epoch=5, save_dir="outputs"):
    """Save comprehensive results to files."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save detailed results as text
    results_file = os.path.join(save_dir, f"{model_name.lower()}_5_epochs_detailed_results.txt")
    
    with open(results_file, 'w') as f:
        f.write(f"🚀 {model_name} QLoRA Fine-tuning Results ({epoch} Epochs)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("📊 OVERALL PERFORMANCE METRICS:\n")
        f.write(f"   • Accuracy:           {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"   • F1-Score (Binary):  {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)\n")
        f.write(f"   • Precision (Binary): {results['precision']:.4f} ({results['precision']*100:.2f}%)\n")
        f.write(f"   • Recall (Binary):    {results['recall']:.4f} ({results['recall']*100:.2f}%)\n")
        f.write(f"   • ROC AUC:            {results['roc_auc']:.4f} ({results['roc_auc']*100:.2f}%)\n")
        f.write(f"   • Average Precision:  {results['average_precision']:.4f} ({results['average_precision']*100:.2f}%)\n\n")
        
        f.write("🎯 CLASS-WISE PERFORMANCE:\n")
        class_0 = results['class_wise_metrics']['class_0']
        class_1 = results['class_wise_metrics']['class_1']
        
        f.write(f"   📋 Class 0 (Negative):\n")
        f.write(f"      - Precision: {class_0['precision']:.4f} ({class_0['precision']*100:.2f}%)\n")
        f.write(f"      - Recall:    {class_0['recall']:.4f} ({class_0['recall']*100:.2f}%)\n")
        f.write(f"      - F1-Score:  {class_0['f1']:.4f} ({class_0['f1']*100:.2f}%)\n")
        f.write(f"      - Support:   {class_0['support']} samples\n\n")
        
        f.write(f"   📋 Class 1 (Positive):\n")
        f.write(f"      - Precision: {class_1['precision']:.4f} ({class_1['precision']*100:.2f}%)\n")
        f.write(f"      - Recall:    {class_1['recall']:.4f} ({class_1['recall']*100:.2f}%)\n")
        f.write(f"      - F1-Score:  {class_1['f1']:.4f} ({class_1['f1']*100:.2f}%)\n")
        f.write(f"      - Support:   {class_1['support']} samples\n\n")
        
        cm = results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        f.write("🔢 CONFUSION MATRIX ANALYSIS:\n")
        f.write(f"   📊 Confusion Matrix:\n")
        f.write(f"      [[{tn:4d}, {fp:4d}]  <- [TN, FP]\n")
        f.write(f"       [{fn:4d}, {tp:4d}]] <- [FN, TP]\n\n")
        
        f.write(f"   📈 Detailed Breakdown:\n")
        total = tn + fp + fn + tp
        f.write(f"      • True Negatives:  {tn:4d} ({tn/total*100:.1f}%)\n")
        f.write(f"      • False Positives: {fp:4d} ({fp/total*100:.1f}%)\n")
        f.write(f"      • False Negatives: {fn:4d} ({fn/total*100:.1f}%)\n")
        f.write(f"      • True Positives:  {tp:4d} ({tp/total*100:.1f}%)\n\n")
        
        f.write(f"   🎯 Additional Metrics:\n")
        f.write(f"      • Specificity (TNR): {results['specificity']:.4f} ({results['specificity']*100:.2f}%)\n")
        f.write(f"      • Sensitivity (TPR): {results['sensitivity']:.4f} ({results['sensitivity']*100:.2f}%)\n")
        f.write(f"      • False Positive Rate: {fp/(fp+tn):.4f} ({fp/(fp+tn)*100:.2f}%)\n")
        f.write(f"      • False Negative Rate: {fn/(fn+tp):.4f} ({fn/(fn+tp)*100:.2f}%)\n")
    
    print(f"📝 Detailed results saved to: {results_file}")
    
    # Save metrics as CSV for further analysis
    csv_file = os.path.join(save_dir, f"{model_name.lower()}_5_epochs_metrics.csv")
    metrics_df = pd.DataFrame([{
        'Model': f"{model_name}_5_epochs",
        'Accuracy': results['accuracy'],
        'F1_Score': results['f1_score'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'ROC_AUC': results['roc_auc'],
        'Average_Precision': results['average_precision'],
        'Specificity': results['specificity'],
        'Sensitivity': results['sensitivity'],
        'True_Negatives': tn,
        'False_Positives': fp,
        'False_Negatives': fn,
        'True_Positives': tp
    }])
    
    metrics_df.to_csv(csv_file, index=False)
    print(f"📊 Metrics CSV saved to: {csv_file}")
    
    return results_file, csv_file

# ==============================================================================
# TRAINING METRICS
# ==============================================================================
def compute_metrics(eval_pred):
    """Compute evaluation metrics for training."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================
def resume_training():
    """Main function to resume training from checkpoint."""
    
    # Setup environment
    device, has_bf16_support = setup_environment()
    if device is None:
        return False
    
    # Load data
    train_dataset, val_dataset, test_dataset = load_and_prepare_data(DATASET_PATH)
    if train_dataset is None:
        return False
    
    # Load model from checkpoint
    model, tokenizer = load_model_from_checkpoint(CHECKPOINT_PATH, device, has_bf16_support)
    if model is None:
        return False
    
    # Tokenize datasets
    print("\\n🔤 Tokenizing datasets...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH
        )
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    print("✅ Datasets tokenized successfully!")
    
    # Configure training arguments for resuming
    print("\\n⚙️ Configuring training arguments for resume...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TOTAL_EPOCHS,  # Total epochs (will resume from where it left off)
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="paged_adamw_8bit",
        fp16=not has_bf16_support,
        bf16=has_bf16_support,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,  # Keep more checkpoints during extended training
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        remove_unused_columns=True,
        dataloader_pin_memory=False,
        push_to_hub=False,
        resume_from_checkpoint=CHECKPOINT_PATH,  # Key parameter for resuming
    )
    
    # Initialize trainer
    print("🏋️ Initializing trainer for resume...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )
    
    print(f"\\n🚀 RESUMING TRAINING FROM CHECKPOINT!")
    print(f"   📊 Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"   📚 Learning rate: {LEARNING_RATE}")
    print(f"   🔄 Target epochs: {TOTAL_EPOCHS} (resume from epoch {COMPLETED_EPOCHS})")
    print(f"   🎯 Current best F1: 97.92% (target: improve further)")
    print(f"   💾 Precision: {'BF16' if has_bf16_support else 'FP16'}")
    
    # Start resumed training
    try:
        trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)
        print("\\n✅ RESUMED TRAINING COMPLETED!")
        
        # Final evaluation on test set with comprehensive metrics
        print("\\n🧪 COMPREHENSIVE FINAL EVALUATION...")
        print("=" * 60)
        
        test_results = trainer.predict(tokenized_test)
        y_pred = np.argmax(test_results.predictions, axis=1)
        y_true = test_results.label_ids
        
        # Get probabilities for positive class (for ROC/PR curves)
        y_proba = torch.softmax(torch.tensor(test_results.predictions), dim=1)[:, 1].numpy()
        
        # Generate comprehensive evaluation report
        detailed_results = generate_comprehensive_report(
            y_true, y_pred, y_proba, model_name="Stella QLoRA", epoch=5
        )
        
        # Create output directory for plots and results
        output_dir = "outputs/stella_5_epochs"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\\n📊 GENERATING VISUALIZATIONS...")
        print("=" * 40)
        
        # Plot confusion matrix
        cm_path = os.path.join(output_dir, "stella_5epochs_confusion_matrix.png")
        cm = plot_confusion_matrix(y_true, y_pred, "Stella QLoRA (5 Epochs)", cm_path)
        
        # Plot ROC curve
        roc_path = os.path.join(output_dir, "stella_5epochs_roc_curve.png")
        roc_auc = plot_roc_curve(y_true, y_proba, "Stella QLoRA (5 Epochs)", roc_path)
        
        # Plot Precision-Recall curve
        pr_path = os.path.join(output_dir, "stella_5epochs_pr_curve.png")
        avg_precision = plot_precision_recall_curve(y_true, y_proba, "Stella QLoRA (5 Epochs)", pr_path)
        
        # Save comprehensive results
        print(f"\\n💾 SAVING COMPREHENSIVE RESULTS...")
        results_file, csv_file = save_results_to_file(
            detailed_results, model_name="Stella_QLoRA", epoch=5, save_dir=output_dir
        )
        
        # Calculate final metrics for comparison
        final_accuracy = detailed_results['accuracy']
        f1 = detailed_results['f1_score']
        precision = detailed_results['precision']
        recall = detailed_results['recall']
        
        print("\\n🏆 FINAL RESULTS SUMMARY (5 EPOCHS):")
        print("=" * 60)
        print(f"📊 Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        print(f"🎯 Test F1-Score: {f1:.4f} ({f1*100:.2f}%)")
        print(f"⚡ Test Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"🔍 Test Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"📈 ROC AUC: {roc_auc:.4f} ({roc_auc*100:.2f}%)")
        print(f"🎯 Average Precision: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
        
        # Compare with previous best (3 epochs)
        previous_best = 0.9792
        improvement = f1 - previous_best
        
        print(f"\\n📈 IMPROVEMENT ANALYSIS:")
        print("=" * 40)
        print(f"   🏅 Previous best (3 epochs): {previous_best:.4f} (97.92%)")
        print(f"   🚀 Current result (5 epochs): {f1:.4f} ({f1*100:.2f}%)")
        print(f"   📊 Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        if improvement > 0.001:  # More than 0.1% improvement
            print("   ✅ Significant improvement with additional training!")
            performance_status = "IMPROVED"
        elif improvement > -0.001:  # Within 0.1% margin
            print("   ⚖️ Performance maintained (within margin of error)")
            performance_status = "MAINTAINED"
        else:
            print("   ⚠️ Performance decreased - consider early stopping for future training")
            performance_status = "DECREASED"
        
        print(f"\\n📋 FILES GENERATED:")
        print("=" * 30)
        print(f"   📊 Confusion Matrix: {cm_path}")
        print(f"   📈 ROC Curve: {roc_path}")
        print(f"   📊 PR Curve: {pr_path}")
        print(f"   📝 Detailed Results: {results_file}")
        print(f"   📊 Metrics CSV: {csv_file}")
        
        # Generate classification report
        print(f"\\n📋 DETAILED CLASSIFICATION REPORT:")
        print("=" * 50)
        print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
        
        # Save final model
        print(f"\\n💾 Saving final model to: {OUTPUT_DIR}")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        return True
        
    except Exception as e:
        print(f"\\n❌ Error during training: {str(e)}")
        return False

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("🚀 STELLA QLORA TRAINING RESUME SCRIPT")
    print("=" * 50)
    
    success = resume_training()
    
    if success:
        print("\\n🎉 TRAINING RESUME COMPLETED SUCCESSFULLY!")
        print("✅ Model trained for full 5 epochs")
        print("📊 Results ready for comparison")
        print("💾 Model saved and ready for inference")
    else:
        print("\\n❌ Training resume failed!")
        print("🔧 Please check the error messages above")
    
    print("\\n" + "=" * 50)
    print("📝 NEXT STEPS:")
    print("   1. 📊 Review comprehensive evaluation results and visualizations")
    print("   2. 📈 Compare 5-epoch vs 3-epoch performance metrics")
    print("   3. � Update model documentation with new results")
    print("   4. � Archive outputs folder with all metrics and plots")
    print("   5. 🎯 Consider ensemble methods or further fine-tuning if needed")
    print("   6. 🚀 Deploy best performing model for production use")

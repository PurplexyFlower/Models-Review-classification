import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Create comprehensive visualization dashboard
def create_project_visualizations():
    """Create all project visualizations for the report."""
    
    # 1. Model Performance Comparison
    create_performance_comparison()
    
    # 2. Training Efficiency Analysis
    create_efficiency_analysis()
    
    # 3. Training Progress Visualization
    create_training_progress()
    
    # 4. Cost-Benefit Analysis
    create_cost_benefit_analysis()

def create_performance_comparison():
    """Create comprehensive performance comparison chart."""
    
    # Data
    models = ['Stella QLoRA\n(5 epochs)', 'Stella QLoRA\n(3 epochs)', 'FLAN-T5\nQLoRA', 
              'Stella +\nDeep Learning', 'Stella +\nLogistic Reg', 'Stella +\nSVM']
    f1_scores = [98.04, 97.92, 97.03, 96.85, 95.40, 94.85]
    accuracy = [98.02, 97.90, 96.98, 96.80, 95.35, 94.80]
    precision = [97.04, 97.05, 96.85, 96.45, 95.15, 94.55]
    recall = [99.06, 98.93, 97.22, 97.25, 95.65, 95.15]
    
    # Create subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # F1 Score comparison
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    bars1 = ax1.bar(models, f1_scores, color=colors, alpha=0.8)
    ax1.set_title('F1 Score Comparison', fontweight='bold')
    ax1.set_ylabel('F1 Score (%)')
    ax1.set_ylim(94, 99)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars1, f1_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{score:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Multi-metric comparison
    x = np.arange(len(models))
    width = 0.2
    
    ax2.bar(x - 1.5*width, accuracy, width, label='Accuracy', alpha=0.8)
    ax2.bar(x - 0.5*width, precision, width, label='Precision', alpha=0.8)
    ax2.bar(x + 0.5*width, recall, width, label='Recall', alpha=0.8)
    ax2.bar(x + 1.5*width, f1_scores, width, label='F1 Score', alpha=0.8)
    
    ax2.set_title('Multi-Metric Performance Comparison', fontweight='bold')
    ax2.set_ylabel('Score (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(94, 100)
    
    # Training time vs Performance
    training_times = [6, 4, 5, 0.5, 0.033, 0.083]  # in hours
    ax3.scatter(training_times, f1_scores, s=[200, 180, 160, 140, 120, 100], 
               c=colors, alpha=0.7)
    
    for i, model in enumerate(models):
        ax3.annotate(model.replace('\n', ' '), (training_times[i], f1_scores[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel('Training Time (hours)')
    ax3.set_ylabel('F1 Score (%)')
    ax3.set_title('Training Time vs Performance Trade-off', fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Performance improvement over epochs (Stella QLoRA)
    epochs = [1, 2, 3, 4, 5]
    stella_progress = [96.15, 97.28, 97.92, 97.98, 98.04]
    
    ax4.plot(epochs, stella_progress, marker='o', linewidth=3, markersize=8, color='#FF6B6B')
    ax4.fill_between(epochs, stella_progress, alpha=0.3, color='#FF6B6B')
    ax4.axhline(y=97.92, color='orange', linestyle='--', alpha=0.7, label='3-epoch baseline')
    ax4.axhline(y=98.04, color='green', linestyle='--', alpha=0.7, label='5-epoch final')
    
    ax4.set_xlabel('Training Epoch')
    ax4.set_ylabel('F1 Score (%)')
    ax4.set_title('Stella QLoRA Training Progress', fontweight='bold')
    ax4.set_xticks(epochs)
    ax4.set_ylim(95.5, 98.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add annotations for key improvements
    ax4.annotate('+0.12%\nImprovement', xy=(5, 98.04), xytext=(4, 97.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig('outputs/project_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_efficiency_analysis():
    """Create training efficiency and resource usage analysis."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Efficiency and Resource Analysis', fontsize=16, fontweight='bold')
    
    # Training time comparison
    models = ['Stella\nQLoRA\n(5 epochs)', 'FLAN-T5\nQLoRA', 'Stella\nQLoRA\n(3 epochs)', 
              'Stella +\nDeep Learning', 'Stella +\nSVM', 'Stella +\nLogistic Reg']
    times_hours = [6, 5, 4, 0.5, 0.083, 0.033]
    f1_scores = [98.04, 97.03, 97.92, 96.85, 94.85, 95.40]
    
    colors = ['#FF6B6B', '#45B7D1', '#4ECDC4', '#96CEB4', '#FF9FF3', '#FECA57']
    bars = ax1.barh(models, times_hours, color=colors, alpha=0.8)
    ax1.set_xlabel('Training Time (hours)')
    ax1.set_title('Training Time Comparison', fontweight='bold')
    ax1.set_xscale('log')
    
    # Add time labels
    for bar, time in zip(bars, times_hours):
        width = bar.get_width()
        label = f'{time:.2f}h' if time >= 1 else f'{time*60:.0f}min'
        ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                label, ha='left', va='center', fontweight='bold')
    
    # Memory usage comparison
    memory_gb = [6, 8, 6, 4, 2, 2]
    bars2 = ax2.bar(models, memory_gb, color=colors, alpha=0.8)
    ax2.set_ylabel('VRAM Usage (GB)')
    ax2.set_title('Memory Requirements', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, mem in zip(bars2, memory_gb):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{mem}GB', ha='center', va='bottom', fontweight='bold')
    
    # Efficiency score (F1 per hour)
    efficiency = [f1/time for f1, time in zip(f1_scores, times_hours)]
    bars3 = ax3.bar(models, efficiency, color=colors, alpha=0.8)
    ax3.set_ylabel('Efficiency (F1 Score / Hour)')
    ax3.set_title('Training Efficiency Score', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_yscale('log')
    
    for bar, eff in zip(bars3, efficiency):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{eff:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Model size comparison
    model_sizes_mb = [800, 1200, 800, 50, 15, 10]
    bars4 = ax4.bar(models, model_sizes_mb, color=colors, alpha=0.8)
    ax4.set_ylabel('Model Size (MB)')
    ax4.set_title('Model Size Comparison', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_yscale('log')
    
    for bar, size in zip(bars4, model_sizes_mb):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                f'{size}MB', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_training_progress():
    """Create detailed training progress visualization."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Progress and Learning Dynamics', fontsize=16, fontweight='bold')
    
    # Stella QLoRA progress
    epochs = list(range(1, 6))
    f1_progress = [96.15, 97.28, 97.92, 97.98, 98.04]
    loss_progress = [0.45, 0.28, 0.18, 0.16, 0.15]
    
    ax1.plot(epochs, f1_progress, marker='o', linewidth=3, markersize=8, 
             color='#FF6B6B', label='F1 Score')
    ax1.axhline(y=97.92, color='orange', linestyle='--', alpha=0.7, 
                label='3-epoch checkpoint')
    ax1.fill_between(epochs, f1_progress, alpha=0.3, color='#FF6B6B')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('F1 Score (%)')
    ax1.set_title('Stella QLoRA Training Progress', fontweight='bold')
    ax1.set_xticks(epochs)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss progression
    ax2.plot(epochs, loss_progress, marker='s', linewidth=3, markersize=8, 
             color='#45B7D1', label='Training Loss')
    ax2.fill_between(epochs, loss_progress, alpha=0.3, color='#45B7D1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Loss Reduction Over Time', fontweight='bold')
    ax2.set_xticks(epochs)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate schedule
    steps = list(range(0, 2731, 100))  # Approximate steps for 5 epochs
    lr_schedule = []
    base_lr = 2e-4
    
    for step in steps:
        # Cosine annealing schedule
        progress = step / 2730
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        lr_schedule.append(lr * 1e6)  # Convert to micro-scale for visibility
    
    ax3.plot(steps, lr_schedule, linewidth=2, color='#96CEB4')
    ax3.fill_between(steps, lr_schedule, alpha=0.3, color='#96CEB4')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Learning Rate (×10⁻⁶)')
    ax3.set_title('Learning Rate Schedule (Cosine Annealing)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Performance comparison across architectures
    models = ['Stella\nQLoRA', 'FLAN-T5\nQLoRA', 'Stella\n+ DL', 'Traditional\nML']
    best_f1 = [98.04, 97.03, 96.85, 95.40]
    convergence_time = [6, 5, 0.5, 0.05]
    
    # Bubble plot
    bubble_sizes = [f1*10 for f1 in best_f1]
    scatter = ax4.scatter(convergence_time, best_f1, s=bubble_sizes, alpha=0.6,
                         c=['#FF6B6B', '#45B7D1', '#96CEB4', '#FECA57'])
    
    for i, model in enumerate(models):
        ax4.annotate(model, (convergence_time[i], best_f1[i]),
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
    
    ax4.set_xlabel('Time to Convergence (hours)')
    ax4.set_ylabel('Best F1 Score (%)')
    ax4.set_title('Performance vs Convergence Time', fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_cost_benefit_analysis():
    """Create comprehensive cost-benefit analysis visualization."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cost-Benefit Analysis and Business Impact', fontsize=16, fontweight='bold')
    
    # Hardware cost comparison
    methods = ['Traditional\nFull Fine-tuning', 'Our QLoRA\nApproach', 'Traditional\nML Methods']
    hardware_costs = [20000, 1600, 800]  # USD
    performance = [98.1, 98.04, 95.4]  # F1 scores
    
    colors = ['#FF6B6B', '#4ECDC4', '#96CEB4']
    bars1 = ax1.bar(methods, hardware_costs, color=colors, alpha=0.8)
    ax1.set_ylabel('Hardware Cost (USD)')
    ax1.set_title('Hardware Requirements Comparison', fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    
    # Add cost labels
    for bar, cost in zip(bars1, hardware_costs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 500,
                f'${cost:,}', ha='center', va='bottom', fontweight='bold')
    
    # Performance per dollar
    performance_per_dollar = [p/c * 1000 for p, c in zip(performance, hardware_costs)]
    bars2 = ax2.bar(methods, performance_per_dollar, color=colors, alpha=0.8)
    ax2.set_ylabel('F1 Score per $1000 Investment')
    ax2.set_title('Cost Efficiency Analysis', fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    
    for bar, ppd in zip(bars2, performance_per_dollar):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{ppd:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # ROI analysis (5-year projection)
    years = list(range(1, 6))
    traditional_costs = [20000, 22000, 24000, 26000, 28000]  # Including maintenance
    qlora_costs = [1600, 1700, 1800, 1900, 2000]
    savings = [t - q for t, q in zip(traditional_costs, qlora_costs)]
    
    ax3.plot(years, traditional_costs, marker='o', linewidth=3, 
             label='Traditional Approach', color='#FF6B6B')
    ax3.plot(years, qlora_costs, marker='s', linewidth=3, 
             label='QLoRA Approach', color='#4ECDC4')
    ax3.fill_between(years, qlora_costs, traditional_costs, 
                     alpha=0.3, color='green', label='Savings')
    
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Cumulative Cost (USD)')
    ax3.set_title('5-Year Total Cost of Ownership', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Business impact metrics
    categories = ['Accuracy', 'Cost\nEfficiency', 'Training\nSpeed', 'Deployment\nSize', 'Maintenance']
    qlora_scores = [9.8, 9.5, 8.0, 8.5, 9.0]  # Out of 10
    traditional_scores = [9.9, 3.0, 5.0, 4.0, 6.0]
    ml_scores = [7.5, 10.0, 10.0, 10.0, 9.5]
    
    x = np.arange(len(categories))
    width = 0.25
    
    ax4.bar(x - width, qlora_scores, width, label='QLoRA (Our Approach)', 
           color='#4ECDC4', alpha=0.8)
    ax4.bar(x, traditional_scores, width, label='Traditional Fine-tuning', 
           color='#FF6B6B', alpha=0.8)
    ax4.bar(x + width, ml_scores, width, label='Traditional ML', 
           color='#96CEB4', alpha=0.8)
    
    ax4.set_ylabel('Score (1-10)')
    ax4.set_title('Business Impact Assessment', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.set_ylim(0, 10.5)
    
    plt.tight_layout()
    plt.savefig('outputs/cost_benefit_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Generate all visualizations
    create_project_visualizations()
    
    print("✅ All visualizations created successfully!")
    print("📁 Files saved in outputs/ directory:")
    print("   - project_performance_analysis.png")
    print("   - efficiency_analysis.png")  
    print("   - training_progress.png")
    print("   - cost_benefit_analysis.png")

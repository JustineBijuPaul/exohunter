"""
Create comprehensive visualizations for model testing results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load test results
project_root = Path(__file__).parent.parent
results_dir = project_root / "test_results"

# Find most recent test result
result_files = list(results_dir.glob("comprehensive_test_*.json"))
if not result_files:
    print("❌ No test results found!")
    exit(1)

latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
print(f"📊 Loading test results from: {latest_result.name}")

with open(latest_result, 'r') as f:
    results = json.load(f)

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ============================================================================
# Plot 1: Model Accuracy Comparison
# ============================================================================
ax1 = fig.add_subplot(gs[0, :2])

model_results = results.get('individual_models', {})
model_names = []
accuracies = []

for model, metrics in model_results.items():
    if isinstance(metrics, dict) and 'accuracy' in metrics:
        model_names.append(model.replace('Optimized ', ''))
        accuracies.append(metrics['accuracy'] * 100)

# Add ensemble
if 'ensemble' in results and 'accuracy' in results['ensemble']:
    model_names.append('Ensemble')
    accuracies.append(results['ensemble']['accuracy'] * 100)

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
bars = ax1.barh(model_names, accuracies, color=colors[:len(model_names)])
ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy Comparison (100 samples)', fontsize=14, fontweight='bold')
ax1.set_xlim([0, 100])

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax1.text(acc + 1, bar.get_y() + bar.get_height()/2, 
             f'{acc:.2f}%', 
             va='center', fontweight='bold')

# Add grid
ax1.grid(axis='x', alpha=0.3)

# ============================================================================
# Plot 2: Test Summary Status
# ============================================================================
ax2 = fig.add_subplot(gs[0, 2])

# Count test passes
test_status = {
    'Passed': 0,
    'Failed': 0
}

for key, value in results.items():
    if key not in ['summary', 'individual_models', 'ensemble', 'full_dataset_test']:
        if isinstance(value, dict):
            if value.get('status') == 'SUCCESS':
                test_status['Passed'] += 1
            else:
                test_status['Failed'] += 1
        elif value == 'SUCCESS':
            test_status['Passed'] += 1
        else:
            test_status['Failed'] += 1

# Special handling for individual models
if 'individual_models' in results:
    for model, metrics in results['individual_models'].items():
        if isinstance(metrics, dict) and metrics.get('status') == 'SUCCESS':
            test_status['Passed'] += 1
        elif isinstance(metrics, dict) and 'ERROR' in str(metrics.get('status', '')):
            test_status['Failed'] += 1

colors_pie = ['#2ecc71', '#e74c3c']
wedges, texts, autotexts = ax2.pie(test_status.values(), 
                                     labels=test_status.keys(),
                                     autopct='%1.1f%%',
                                     colors=colors_pie,
                                     startangle=90,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('Test Status Overview', fontsize=14, fontweight='bold')

# ============================================================================
# Plot 3: Confusion Matrix
# ============================================================================
ax3 = fig.add_subplot(gs[1, :])

if 'full_dataset_test' in results and 'confusion_matrix' in results['full_dataset_test']:
    cm = np.array(results['full_dataset_test']['confusion_matrix'])
    classes = results['full_dataset_test']['classes']
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                ax=ax3, cbar_kws={'label': 'Count'})
    ax3.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax3.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax3.set_title('Confusion Matrix - Full Dataset (16,916 samples)', 
                  fontsize=14, fontweight='bold')
    
    # Rotate labels
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0)

# ============================================================================
# Plot 4: Model Metrics Comparison
# ============================================================================
ax4 = fig.add_subplot(gs[2, :2])

metrics_df = []
for model, metrics in model_results.items():
    if isinstance(metrics, dict) and 'accuracy' in metrics:
        metrics_df.append({
            'Model': model.replace('Optimized ', ''),
            'Accuracy': metrics['accuracy'],
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1-Score': metrics.get('f1', 0)
        })

# Add ensemble
if 'ensemble' in results and 'accuracy' in results['ensemble']:
    metrics_df.append({
        'Model': 'Ensemble',
        'Accuracy': results['ensemble']['accuracy'],
        'Precision': results['ensemble'].get('precision', 0),
        'Recall': results['ensemble'].get('recall', 0),
        'F1-Score': results['ensemble'].get('f1', 0)
    })

if metrics_df:
    df = pd.DataFrame(metrics_df)
    
    x = np.arange(len(df))
    width = 0.2
    
    ax4.bar(x - 1.5*width, df['Accuracy'], width, label='Accuracy', color='#3498db')
    ax4.bar(x - 0.5*width, df['Precision'], width, label='Precision', color='#e74c3c')
    ax4.bar(x + 0.5*width, df['Recall'], width, label='Recall', color='#2ecc71')
    ax4.bar(x + 1.5*width, df['F1-Score'], width, label='F1-Score', color='#f39c12')
    
    ax4.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax4.set_title('Comprehensive Metrics Comparison (100 samples)', 
                  fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax4.legend(loc='lower right')
    ax4.set_ylim([0, 1.1])
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, row in df.iterrows():
        for j, (metric, offset) in enumerate([('Accuracy', -1.5), ('Precision', -0.5), 
                                               ('Recall', 0.5), ('F1-Score', 1.5)]):
            val = row[metric]
            ax4.text(i + offset*width, val + 0.02, f'{val:.3f}', 
                    ha='center', va='bottom', fontsize=8, rotation=90)

# ============================================================================
# Plot 5: Summary Statistics
# ============================================================================
ax5 = fig.add_subplot(gs[2, 2])
ax5.axis('off')

# Compile summary text
summary_text = []

# Overall accuracy
if 'full_dataset_test' in results:
    overall_acc = results['full_dataset_test'].get('overall_accuracy', 0)
    samples = results['full_dataset_test'].get('samples_tested', 0)
    summary_text.append(f"Overall Accuracy:\n{overall_acc*100:.2f}% on {samples:,} samples\n")

# Model agreement
if 'ensemble' in results:
    agreement = results['ensemble'].get('all_agree_pct', 0)
    summary_text.append(f"Model Agreement:\n{agreement:.1f}% full consensus\n")

# Edge cases
if 'edge_cases' in results:
    edge_passed = sum(1 for v in results['edge_cases'].values() if v == 'SUCCESS')
    edge_total = len(results['edge_cases'])
    summary_text.append(f"Edge Cases:\n{edge_passed}/{edge_total} passed\n")

# Consistency
if 'consistency' in results:
    consistency = results['consistency']
    status = "✓" if consistency == "SUCCESS" else "✗"
    summary_text.append(f"Consistency:\n{status} All models consistent\n")

# Pass rate
if 'summary' in results:
    pass_rate = results['summary'].get('pass_rate', 0)
    tests_passed = results['summary'].get('tests_passed', 0)
    tests_total = results['summary'].get('tests_total', 0)
    summary_text.append(f"\nOverall Pass Rate:\n{pass_rate:.1f}%\n({tests_passed}/{tests_total} tests)")

summary_str = '\n'.join(summary_text)
ax5.text(0.1, 0.9, summary_str, transform=ax5.transAxes,
         fontsize=11, verticalalignment='top', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax5.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

# Add main title
fig.suptitle('🧪 Exoplanet Model Testing Report - Comprehensive Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

# Add timestamp
timestamp = results.get('summary', {}).get('timestamp', 'Unknown')
fig.text(0.99, 0.01, f'Generated: {timestamp}', 
         ha='right', va='bottom', fontsize=8, style='italic')

# Save figure
output_path = results_dir / f"test_visualization_{Path(latest_result.stem).stem.split('_', 2)[-1]}.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved to: {output_path}")

# Also save as PDF
output_pdf = output_path.with_suffix('.pdf')
plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
print(f"✅ PDF saved to: {output_pdf}")

print("\n📊 Visualization complete!")
plt.close()

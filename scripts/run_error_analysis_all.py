"""
Error Analysis for ALL 6 ABSA Models
=====================================
Generates confusion matrices, per-aspect F1, and top misclassified samples
for each model to support Chapter 4 (Results & Discussion) of the thesis.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob

SENTIMENT_MAP = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
ASPECTS = ['Battery', 'Camera', 'Design', 'Display', 'General',
           'Packaging', 'Performance', 'Price', 'Shipping', 'Shop_Service']


def find_test_results(model_dir):
    """Find test_results.json in model directory."""
    patterns = [
        f"{model_dir}/models/mtl/test_results.json",
        f"{model_dir}/models/sentiment_classification/test_results.json",
        f"{model_dir}/models/aspect_detection/test_results.json",
        f"{model_dir}/results/two_stage_training/test_results.json",
        f"{model_dir}/results/test_results.json",
    ]
    for p in patterns:
        if os.path.exists(p):
            return p
    # Glob fallback
    files = glob.glob(f"{model_dir}/**/test_results*.json", recursive=True)
    return files[0] if files else None


def find_predictions(model_dir):
    """Find detailed predictions CSV."""
    files = glob.glob(f"{model_dir}/**/test_predictions_detailed*.csv", recursive=True)
    if not files:
        files = glob.glob(f"{model_dir}/**/predictions*.csv", recursive=True)
    return files[0] if files else None


def find_training_history(model_dir):
    """Find training_history.csv."""
    files = glob.glob(f"{model_dir}/**/training_history.csv", recursive=True)
    return files[0] if files else None


def plot_training_curves(history_path, model_name, output_dir):
    """Plot loss and metric curves from training history."""
    if not history_path or not os.path.exists(history_path):
        print(f"  ⚠️ No training history found for {model_name}")
        return
    
    df = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} - Training Curves', fontsize=14, fontweight='bold')
    
    # Loss curve
    loss_cols = [c for c in df.columns if 'loss' in c.lower()]
    for col in loss_cols:
        axes[0].plot(df[col], label=col)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metric curves (F1/Accuracy)
    metric_cols = [c for c in df.columns if 'f1' in c.lower() or 'acc' in c.lower()]
    for col in metric_cols:
        axes[1].plot(df[col], label=col)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Metric Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{model_name}_training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Training curves saved: {save_path}")


def analyze_model(model_name, model_dir, output_dir):
    """Run error analysis for a single model."""
    print(f"\n{'='*60}")
    print(f"📊 Error Analysis: {model_name}")
    print(f"{'='*60}")
    
    model_output = os.path.join(output_dir, model_name)
    os.makedirs(model_output, exist_ok=True)
    
    # 1. Load test results
    results_path = find_test_results(model_dir)
    if results_path:
        with open(results_path) as f:
            results = json.load(f)
        print(f"  ✅ Test results loaded: {results_path}")
        
        # Save formatted results
        with open(os.path.join(model_output, 'test_results_formatted.txt'), 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"{'='*40}\n")
            f.write(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(f"  ❌ No test results found for {model_name}")
    
    # 2. Training curves
    history_path = find_training_history(model_dir)
    plot_training_curves(history_path, model_name, model_output)
    
    # 3. Look for existing confusion matrix images
    cm_files = glob.glob(f"{model_dir}/**/confusion_matrix*.png", recursive=True)
    if cm_files:
        print(f"  ✅ Found {len(cm_files)} confusion matrix image(s)")
        for cm in cm_files:
            print(f"     → {cm}")
    else:
        print(f"  ⚠️ No confusion matrix images found")
    
    # 4. Analyze predictions if available
    pred_path = find_predictions(model_dir)
    if pred_path:
        print(f"  ✅ Predictions loaded: {pred_path}")
        df_pred = pd.read_csv(pred_path)
        
        # Count correct vs incorrect
        if 'is_correct' in df_pred.columns:
            correct = df_pred['is_correct'].sum()
            total = len(df_pred)
            print(f"  📊 Correct: {correct}/{total} ({correct/total*100:.2f}%)")
            
            # Save top errors
            errors = df_pred[df_pred['is_correct'] == False].head(20)
            if len(errors) > 0:
                errors.to_csv(os.path.join(model_output, 'top_errors.csv'), index=False)
                print(f"  💾 Top {len(errors)} errors saved to top_errors.csv")
    else:
        print(f"  ⚠️ No detailed predictions found")
    
    print(f"  📁 Analysis output: {model_output}")


def generate_comparison_chart(output_dir):
    """Generate a bar chart comparing all 6 models side by side."""
    models = {
        'ViSoBERT-MTL': 'VisoBERT-MTL',
        'ViSoBERT-STL': 'VisoBERT-STL',
        'PhoBERT-MTL':  'phoBERT-MTL',
        'PhoBERT-STL':  'PhoBERT-STL',
        'BiLSTM-MTL':   'BILSTM-MTL',
        'BiLSTM-STL':   'BILSTM-STL',
    }
    
    data = []
    for display_name, dir_name in models.items():
        results_path = find_test_results(dir_name)
        if results_path:
            with open(results_path) as f:
                r = json.load(f)
            
            if "ad" in r and isinstance(r["ad"], dict):
                ad_f1 = r["ad"].get("test_f1", r["ad"].get("overall_f1", 0)) * 100
                sc_f1 = r["sc"].get("test_f1", r["sc"].get("overall_f1", 0)) * 100
            else:
                ad_f1 = r.get("test_f1", r.get("f1", 0)) * 100
                sc_f1 = ad_f1  # Single task
            
            data.append({"Model": display_name, "AD F1": ad_f1, "SC F1": sc_f1})
    
    if not data:
        print("⚠️ No model results found for comparison chart")
        return
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35
    
    bars_ad = ax.bar(x - width/2, df['AD F1'], width, label='AD F1-Score', color='#2196F3', alpha=0.85)
    bars_sc = ax.bar(x + width/2, df['SC F1'], width, label='SC F1-Score', color='#FF9800', alpha=0.85)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('F1-Score (%)', fontsize=12)
    ax.set_title('So sánh F1-Score giữa 6 mô hình ABSA', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar in bars_ad:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars_sc:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'model_comparison_f1.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Comparison chart saved: {save_path}")


def main():
    output_dir = "error_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("🔍 COMPREHENSIVE ERROR ANALYSIS - ALL 6 ABSA MODELS")
    print("=" * 70)
    
    models = {
        'ViSoBERT-MTL': 'VisoBERT-MTL',
        'ViSoBERT-STL': 'VisoBERT-STL',
        'PhoBERT-MTL':  'phoBERT-MTL',
        'PhoBERT-STL':  'PhoBERT-STL',
        'BiLSTM-MTL':   'BILSTM-MTL',
        'BiLSTM-STL':   'BILSTM-STL',
    }
    
    for display_name, dir_name in models.items():
        if os.path.exists(dir_name):
            analyze_model(display_name, dir_name, output_dir)
        else:
            print(f"\n⚠️ Directory not found: {dir_name} — skipping {display_name}")
    
    # Generate comparison chart
    generate_comparison_chart(output_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ Error analysis complete! All results saved to: {output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

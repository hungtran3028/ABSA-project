"""
Error Analysis for ABSA Models
===============================
Analyzes prediction errors across all 6 models.
- Top-20 misclassified samples per model
- Error pattern analysis (which aspects are hardest)
- Cross-model error agreement

Usage:
    python scripts/run_error_analysis_all.py --results_dir results/ABSA-results
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from collections import Counter


ASPECT_NAMES = [
    'Battery', 'Camera', 'Performance', 'Display', 'Design',
    'Packaging', 'Price', 'Shop_Service', 'Shipping', 'General', 'Others'
]

MODELS = {
    'ViSoBERT-MTL': 'ViSoBERT-MTL/test_predictions_detailed.csv',
    'PhoBERT-MTL': 'PhoBERT-MTL/test_predictions_detailed.csv',
    'BiLSTM-MTL': 'BiLSTM-MTL/test_predictions_detailed.csv',
    'ViSoBERT-STL': 'ViSoBERT-STL/sentiment_classification/test_predictions_detailed.csv',
    'PhoBERT-STL': 'PhoBERT-STL/sentiment_classification/test_predictions_detailed.csv',
    'BiLSTM-STL': 'BiLSTM-STL/sentiment_classification/test_predictions_detailed.csv',
}


def analyze_mtl_errors(df, model_name):
    """Analyze errors for MTL model format."""
    results = {'model': model_name, 'type': 'mtl'}
    
    # AD errors per aspect
    ad_errors = {}
    for aspect in ASPECT_NAMES:
        pred_col = f'{aspect}_ad_pred'
        true_col = f'{aspect}_ad_true'
        if pred_col in df.columns and true_col in df.columns:
            errors = (df[pred_col] != df[true_col]).sum()
            total = len(df)
            ad_errors[aspect] = {'errors': int(errors), 'total': total, 'error_rate': errors/total}
    results['ad_errors_per_aspect'] = ad_errors
    
    # SC errors per aspect (only where aspect is present)
    sc_errors = {}
    for aspect in ASPECT_NAMES:
        correct_col = f'{aspect}_sc_correct'
        true_col = f'{aspect}_ad_true'
        if correct_col in df.columns and true_col in df.columns:
            mask = df[true_col] == 1
            aspect_df = df[mask]
            if len(aspect_df) > 0:
                valid = aspect_df[correct_col].apply(lambda x: x != '' and not pd.isna(x))
                errors = aspect_df[valid][correct_col].apply(lambda x: float(x) == 0).sum()
                sc_errors[aspect] = {'errors': int(errors), 'total': int(valid.sum()),
                                    'error_rate': errors/valid.sum() if valid.sum() > 0 else 0}
    results['sc_errors_per_aspect'] = sc_errors
    
    # Top misclassified samples (by number of wrong aspects)
    if 'ad_exact_match' in df.columns:
        wrong_samples = df[df['ad_exact_match'] == 0].copy()
        # Count wrong aspects per sample
        wrong_counts = []
        for _, row in wrong_samples.iterrows():
            n_wrong = sum(1 for a in ASPECT_NAMES
                         if f'{a}_ad_correct' in df.columns and row.get(f'{a}_ad_correct', 1) == 0)
            wrong_counts.append(n_wrong)
        wrong_samples = wrong_samples.copy()
        wrong_samples['n_wrong_aspects'] = wrong_counts
        top20 = wrong_samples.nlargest(20, 'n_wrong_aspects')
        results['top20_ad_errors'] = top20[['sample_id', 'n_wrong_aspects']].to_dict('records')
    
    return results


def analyze_stl_errors(df, model_name):
    """Analyze errors for STL model format."""
    results = {'model': model_name, 'type': 'stl'}
    
    # Detect column format
    pred_cols = [c for c in df.columns if c.endswith('_pred')]
    correct_cols = [c for c in df.columns if c.endswith('_correct')]
    
    errors_per_aspect = {}
    for col in pred_cols:
        aspect = col.replace('_pred', '')
        true_col = f'{aspect}_true'
        correct_col = f'{aspect}_correct'
        
        if true_col in df.columns:
            valid = df[true_col].notna() & (df[true_col] != '')
            valid_df = df[valid]
            if len(valid_df) > 0:
                if correct_col in df.columns:
                    valid_correct = valid_df[correct_col].apply(
                        lambda x: x != '' and not pd.isna(x))
                    errors = valid_df[valid_correct][correct_col].apply(
                        lambda x: float(x) == 0).sum()
                else:
                    errors = (valid_df[col].astype(float) != valid_df[true_col].astype(float)).sum()
                errors_per_aspect[aspect] = {
                    'errors': int(errors),
                    'total': int(len(valid_df)),
                    'error_rate': errors / len(valid_df) if len(valid_df) > 0 else 0
                }
    
    results['errors_per_aspect'] = errors_per_aspect
    return results


def main():
    parser = argparse.ArgumentParser(description='Error Analysis for ABSA models')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    results_dir = args.results_dir
    output_dir = args.output_dir or os.path.join(results_dir, 'error_analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Error Analysis for ABSA Models")
    print("=" * 80)
    
    all_results = []
    
    for model_name, pred_path in MODELS.items():
        fpath = os.path.join(results_dir, pred_path)
        if not os.path.exists(fpath):
            print(f"  SKIP {model_name}: {fpath} not found")
            continue
        
        print(f"\n--- {model_name} ---")
        df = pd.read_csv(fpath)
        print(f"  Loaded {len(df)} samples")
        
        if 'ad_exact_match' in df.columns:
            result = analyze_mtl_errors(df, model_name)
        else:
            result = analyze_stl_errors(df, model_name)
        
        all_results.append(result)
        
        # Print summary
        if 'ad_errors_per_aspect' in result:
            top3 = sorted(result['ad_errors_per_aspect'].items(),
                         key=lambda x: x[1]['error_rate'], reverse=True)[:3]
            parts = [f"{a}({v['error_rate']*100:.1f}%)" for a, v in top3]
            print(f"  Hardest AD aspects: {', '.join(parts)}")
        if 'sc_errors_per_aspect' in result:
            top3 = sorted(result['sc_errors_per_aspect'].items(),
                         key=lambda x: x[1]['error_rate'], reverse=True)[:3]
            parts = [f"{a}({v['error_rate']*100:.1f}%)" for a, v in top3]
            print(f"  Hardest SC aspects: {', '.join(parts)}")
        if 'errors_per_aspect' in result:
            top3 = sorted(result['errors_per_aspect'].items(),
                         key=lambda x: x[1]['error_rate'], reverse=True)[:3]
            parts = [f"{a}({v['error_rate']*100:.1f}%)" for a, v in top3]
            print(f"  Hardest aspects: {', '.join(parts)}")
    
    # Save results
    json_path = os.path.join(output_dir, 'error_analysis_detailed.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")
    
    # Generate text report
    txt_path = os.path.join(output_dir, 'error_analysis_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Error Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        for result in all_results:
            f.write(f"Model: {result['model']} ({result['type'].upper()})\n")
            f.write("-" * 40 + "\n")
            
            for task_key in ['ad_errors_per_aspect', 'sc_errors_per_aspect', 'errors_per_aspect']:
                if task_key in result:
                    task_label = task_key.replace('_per_aspect', '').replace('_errors', ' errors').upper()
                    f.write(f"\n  {task_label}:\n")
                    sorted_aspects = sorted(result[task_key].items(),
                                           key=lambda x: x[1]['error_rate'], reverse=True)
                    for aspect, vals in sorted_aspects:
                        f.write(f"    {aspect:<15} {vals['errors']:>4}/{vals['total']:<4} "
                               f"(error rate: {vals['error_rate']*100:.1f}%)\n")
            
            if 'top20_ad_errors' in result:
                f.write(f"\n  Top-20 AD error samples:\n")
                for item in result['top20_ad_errors']:
                    f.write(f"    Sample {item['sample_id']}: {item['n_wrong_aspects']} wrong aspects\n")
            
            f.write("\n")
    
    print(f"Saved: {txt_path}")


if __name__ == '__main__':
    main()

"""
McNemar's Test for Statistical Significance
============================================
Compare model pairs on sample-level correctness using McNemar's test.
Supports both MTL format (single file) and STL format (separate AD/SC files).

Usage:
    python scripts/run_mcnemar_test.py --results_dir results/ABSA-results
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from statsmodels.stats.contingency_tables import mcnemar
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

MODELS = {
    'ViSoBERT-MTL': {
        'type': 'mtl',
        'pred_file': 'ViSoBERT-MTL/test_predictions_detailed.csv',
    },
    'PhoBERT-MTL': {
        'type': 'mtl',
        'pred_file': 'PhoBERT-MTL/test_predictions_detailed.csv',
    },
    'BiLSTM-MTL': {
        'type': 'mtl',
        'pred_file': 'BiLSTM-MTL/test_predictions_detailed.csv',
    },
    'ViSoBERT-STL': {
        'type': 'stl',
        'ad_file': 'ViSoBERT-STL/aspect_detection/test_predictions.csv',
        'sc_file': 'ViSoBERT-STL/sentiment_classification/test_predictions_detailed.csv',
    },
    'PhoBERT-STL': {
        'type': 'stl',
        'ad_file': 'PhoBERT-STL/aspect_detection/test_predictions.csv',
        'sc_file': 'PhoBERT-STL/sentiment_classification/test_predictions_detailed.csv',
    },
    'BiLSTM-STL': {
        'type': 'stl',
        'ad_file': 'BiLSTM-STL/aspect_detection/test_predictions.csv',
        'sc_file': 'BiLSTM-STL/sentiment_classification/test_predictions_detailed.csv',
    },
}

# Pairs to compare (scientific comparisons for thesis)
COMPARISON_PAIRS = [
    # MTL vs STL within same backbone
    ('ViSoBERT-MTL', 'ViSoBERT-STL'),
    ('PhoBERT-MTL', 'PhoBERT-STL'),
    ('BiLSTM-MTL', 'BiLSTM-STL'),
    # Cross-backbone within same paradigm
    ('ViSoBERT-MTL', 'PhoBERT-MTL'),
    ('ViSoBERT-STL', 'PhoBERT-STL'),
    # Best Transformer vs BiLSTM
    ('ViSoBERT-MTL', 'BiLSTM-MTL'),
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_mtl_correctness(filepath):
    """Load MTL predictions: returns (ad_correct_per_sample, sc_correct_per_sample)."""
    df = pd.read_csv(filepath)
    ad_correct = df['ad_exact_match'].values.astype(int)
    sc_correct = df['sc_exact_match'].values.astype(int)
    return ad_correct, sc_correct


def load_stl_ad_correctness(filepath):
    """Load STL AD predictions: returns ad_correct_per_sample."""
    df = pd.read_csv(filepath)
    if 'ad_exact_match' in df.columns:
        return df['ad_exact_match'].values.astype(int)
    # Fallback: compute from per-aspect columns
    correct_cols = [c for c in df.columns if c.endswith('_correct')]
    if correct_cols:
        return (df[correct_cols].sum(axis=1) == len(correct_cols)).astype(int).values
    # Last fallback: compute from pred/true columns
    aspects = [c.replace('_pred', '') for c in df.columns if c.endswith('_pred')]
    all_correct = np.ones(len(df), dtype=int)
    for asp in aspects:
        pred_col = f'{asp}_pred'
        true_col = f'{asp}_true'
        if pred_col in df.columns and true_col in df.columns:
            all_correct &= (df[pred_col] == df[true_col]).astype(int).values
    return all_correct


def load_stl_sc_correctness(filepath):
    """Load STL SC predictions: returns sc_correct_per_sample."""
    df = pd.read_csv(filepath)
    if 'sc_exact_match' in df.columns:
        return df['sc_exact_match'].values.astype(int)
    # Compute from per-aspect correct columns
    correct_cols = [c for c in df.columns if c.endswith('_correct')]
    if correct_cols:
        # For each sample, check if all non-empty correct values are 1
        def row_correct(row):
            vals = [row[c] for c in correct_cols if row[c] != '' and not pd.isna(row[c])]
            if not vals:
                return 1  # No aspects to check
            return int(all(float(v) == 1.0 for v in vals))
        return df.apply(row_correct, axis=1).values.astype(int)
    # Fallback
    aspects = [c.replace('_pred', '') for c in df.columns if c.endswith('_pred')]
    all_correct = np.ones(len(df), dtype=int)
    for asp in aspects:
        pred_col = f'{asp}_pred'
        true_col = f'{asp}_true'
        if pred_col in df.columns and true_col in df.columns:
            all_correct &= (df[pred_col] == df[true_col]).astype(int).values
    return all_correct


def load_model_correctness(model_name, results_dir, task='both'):
    """Load correctness arrays for a model.
    
    Returns dict with keys 'ad' and/or 'sc', each containing a binary array.
    """
    info = MODELS[model_name]
    result = {}
    
    if info['type'] == 'mtl':
        fpath = os.path.join(results_dir, info['pred_file'])
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found!")
            return None
        ad, sc = load_mtl_correctness(fpath)
        result['ad'] = ad
        result['sc'] = sc
    else:  # stl
        if task in ('both', 'ad'):
            ad_path = os.path.join(results_dir, info['ad_file'])
            if os.path.exists(ad_path):
                result['ad'] = load_stl_ad_correctness(ad_path)
            else:
                print(f"  WARNING: {ad_path} not found!")
        if task in ('both', 'sc'):
            sc_path = os.path.join(results_dir, info['sc_file'])
            if os.path.exists(sc_path):
                result['sc'] = load_stl_sc_correctness(sc_path)
            else:
                print(f"  WARNING: {sc_path} not found!")
    
    return result if result else None


# ============================================================================
# MCNEMAR'S TEST
# ============================================================================

def run_mcnemar(correct_a, correct_b, model_a, model_b, task_name):
    """Run McNemar's test on two binary correctness arrays.
    
    Returns dict with contingency table and test results.
    """
    n = min(len(correct_a), len(correct_b))
    a = correct_a[:n]
    b = correct_b[:n]
    
    # Build 2x2 contingency table
    # [both_correct, a_correct_b_wrong]
    # [a_wrong_b_correct, both_wrong]
    both_correct = int(np.sum((a == 1) & (b == 1)))
    a_right_b_wrong = int(np.sum((a == 1) & (b == 0)))
    a_wrong_b_right = int(np.sum((a == 0) & (b == 1)))
    both_wrong = int(np.sum((a == 0) & (b == 0)))
    
    table = np.array([[both_correct, a_right_b_wrong],
                      [a_wrong_b_right, both_wrong]])
    
    result = {
        'model_a': model_a,
        'model_b': model_b,
        'task': task_name,
        'n_samples': n,
        'contingency_table': table.tolist(),
        'both_correct': both_correct,
        'a_right_b_wrong': a_right_b_wrong,
        'a_wrong_b_right': a_wrong_b_right,
        'both_wrong': both_wrong,
    }
    
    discordant = a_right_b_wrong + a_wrong_b_right
    
    if discordant == 0:
        result['statistic'] = 0.0
        result['p_value'] = 1.0
        result['note'] = 'No discordant pairs'
    elif HAS_STATSMODELS:
        # Use exact test if discordant < 25, else chi-square
        use_exact = discordant < 25
        test_result = mcnemar(table, exact=use_exact, correction=True)
        result['statistic'] = float(test_result.statistic)
        result['p_value'] = float(test_result.pvalue)
        result['exact'] = use_exact
    else:
        # Manual chi-square approximation with continuity correction
        b_val = a_right_b_wrong
        c_val = a_wrong_b_right
        chi2 = (abs(b_val - c_val) - 1) ** 2 / (b_val + c_val) if (b_val + c_val) > 0 else 0
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=1)
        result['statistic'] = float(chi2)
        result['p_value'] = float(p_value)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="McNemar's Test for ABSA Models")
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to ABSA-results directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: results_dir/error_analysis_results)')
    args = parser.parse_args()
    
    results_dir = args.results_dir
    output_dir = args.output_dir or os.path.join(results_dir, 'error_analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("McNemar's Statistical Significance Test")
    print("=" * 80)
    
    if not HAS_STATSMODELS:
        print("WARNING: statsmodels not installed. Using manual chi-square.")
    
    # Run comparisons
    all_results = []
    
    for model_a, model_b in COMPARISON_PAIRS:
        print(f"\n--- {model_a} vs {model_b} ---")
        
        data_a = load_model_correctness(model_a, results_dir)
        data_b = load_model_correctness(model_b, results_dir)
        
        if data_a is None or data_b is None:
            print(f"  SKIP: Missing predictions for one or both models")
            continue
        
        for task in ['ad', 'sc']:
            if task not in data_a or task not in data_b:
                print(f"  SKIP {task.upper()}: predictions not available")
                continue
            
            result = run_mcnemar(data_a[task], data_b[task], model_a, model_b,
                               'Aspect Detection' if task == 'ad' else 'Sentiment Classification')
            all_results.append(result)
            
            sig = "✓ Significant" if result['p_value'] < 0.05 else "✗ Not significant"
            print(f"  {task.upper()}: χ²={result['statistic']:.4f}, p={result['p_value']:.6f} → {sig}")
            print(f"       Discordant: {model_a}✓/{model_b}✗={result['a_right_b_wrong']}, "
                  f"{model_a}✗/{model_b}✓={result['a_wrong_b_right']}")
    
    if not all_results:
        print("\nNo comparisons could be made. Check prediction files.")
        return
    
    # Bonferroni correction
    n_tests = len(all_results)
    bonferroni_alpha = 0.05 / n_tests
    print(f"\n{'='*80}")
    print(f"Bonferroni Correction: α = 0.05 / {n_tests} = {bonferroni_alpha:.6f}")
    print(f"{'='*80}")
    
    for r in all_results:
        r['bonferroni_significant'] = r['p_value'] < bonferroni_alpha
        r['bonferroni_alpha'] = bonferroni_alpha
    
    # Save JSON
    json_path = os.path.join(output_dir, 'mcnemar_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved JSON: {json_path}")
    
    # Save text report
    txt_path = os.path.join(output_dir, 'mcnemar_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("McNemar's Test Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of comparisons: {n_tests}\n")
        f.write(f"Bonferroni-corrected α: {bonferroni_alpha:.6f}\n\n")
        
        for r in all_results:
            f.write(f"{r['model_a']} vs {r['model_b']} ({r['task']})\n")
            f.write(f"  Samples: {r['n_samples']}\n")
            f.write(f"  χ² = {r['statistic']:.4f}, p = {r['p_value']:.6f}\n")
            f.write(f"  Discordant: {r['a_right_b_wrong']} / {r['a_wrong_b_right']}\n")
            sig = "YES" if r['bonferroni_significant'] else "NO"
            f.write(f"  Significant (Bonferroni): {sig}\n\n")
    
    print(f"Saved TXT: {txt_path}")
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Pair':<35} {'Task':<5} {'p-value':<12} {'Sig?'}")
    print("-" * 60)
    for r in all_results:
        pair = f"{r['model_a']} vs {r['model_b']}"
        task = 'AD' if 'Aspect' in r['task'] else 'SC'
        sig = "YES*" if r['bonferroni_significant'] else ("yes" if r['p_value'] < 0.05 else "no")
        print(f"{pair:<35} {task:<5} {r['p_value']:<12.6f} {sig}")
    
    print(f"\n* = significant after Bonferroni correction (α={bonferroni_alpha:.6f})")


if __name__ == '__main__':
    main()

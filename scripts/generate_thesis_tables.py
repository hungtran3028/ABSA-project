"""
Generate LaTeX Tables for Thesis Chapter 4
==========================================
Reads test_results.json from all 6 models and produces 3 LaTeX tables:
  1. Overall performance (AD + SC F1/Precision/Recall for all models)
  2. Per-aspect F1 comparison
  3. McNemar's test results

Usage:
    python scripts/generate_thesis_tables.py --results_dir results/ABSA-results
"""

import os
import json
import argparse
import glob


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

MODELS_ORDER = [
    ('ViSoBERT-STL', 'ViSoBERT-STL'),
    ('ViSoBERT-MTL', 'ViSoBERT-MTL'),
    ('PhoBERT-STL', 'PhoBERT-STL'),
    ('PhoBERT-MTL', 'PhoBERT-MTL'),
    ('BiLSTM-STL', 'BiLSTM-STL'),
    ('BiLSTM-MTL', 'BiLSTM-MTL'),
]

# Path patterns to find test_results.json
RESULTS_PATHS = {
    'ViSoBERT-STL': {
        'ad': 'ViSoBERT-STL/aspect_detection/test_results.json',
        'sc': 'ViSoBERT-STL/sentiment_classification/test_results.json',
    },
    'ViSoBERT-MTL': {
        'combined': 'ViSoBERT-MTL/test_results.json',
    },
    'PhoBERT-STL': {
        'ad': 'PhoBERT-STL/aspect_detection/test_results.json',
        'sc': 'PhoBERT-STL/sentiment_classification/test_results.json',
    },
    'PhoBERT-MTL': {
        'combined': 'PhoBERT-MTL/test_results.json',
    },
    'BiLSTM-STL': {
        'ad': 'BiLSTM-STL/aspect_detection/test_results.json',
        'sc': 'BiLSTM-STL/sentiment_classification/test_results.json',
    },
    'BiLSTM-MTL': {
        'combined': 'BiLSTM-MTL/test_results.json',
    },
}


def load_model_metrics(results_dir):
    """Load metrics for all models."""
    all_metrics = {}
    
    for model_name, paths in RESULTS_PATHS.items():
        metrics = {'ad': {}, 'sc': {}}
        
        if 'combined' in paths:
            fpath = os.path.join(results_dir, paths['combined'])
            if os.path.exists(fpath):
                with open(fpath) as f:
                    data = json.load(f)
                metrics['ad'] = data.get('ad', {})
                metrics['sc'] = data.get('sc', {})
        else:
            for task in ['ad', 'sc']:
                fpath = os.path.join(results_dir, paths[task])
                if os.path.exists(fpath):
                    with open(fpath) as f:
                        metrics[task] = json.load(f)
        
        all_metrics[model_name] = metrics
    
    return all_metrics


def fmt(val, pct=True):
    """Format a metric value."""
    if val is None or val == 0:
        return '---'
    if pct:
        return f'{val*100:.2f}'
    return f'{val:.4f}'


def generate_table1_overall(all_metrics, output_dir):
    """Table 1: Overall AD + SC performance for all models."""
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Kết quả tổng quan của 6 mô hình ABSA trên tập test}')
    lines.append(r'\label{tab:overall_results}')
    lines.append(r'\begin{tabular}{l|ccc|ccc}')
    lines.append(r'\hline')
    lines.append(r'\multirow{2}{*}{\textbf{Model}} & \multicolumn{3}{c|}{\textbf{Aspect Detection}} & \multicolumn{3}{c}{\textbf{Sentiment Classification}} \\')
    lines.append(r' & F1 & Precision & Recall & F1 & Precision & Recall \\')
    lines.append(r'\hline')
    
    for model_name, _ in MODELS_ORDER:
        m = all_metrics.get(model_name, {})
        ad = m.get('ad', {})
        sc = m.get('sc', {})
        
        ad_f1 = ad.get('test_f1', ad.get('overall_f1', 0))
        ad_p = ad.get('test_precision', ad.get('overall_precision', 0))
        ad_r = ad.get('test_recall', ad.get('overall_recall', 0))
        sc_f1 = sc.get('test_f1', sc.get('overall_f1', 0))
        sc_p = sc.get('test_precision', sc.get('overall_precision', 0))
        sc_r = sc.get('test_recall', sc.get('overall_recall', 0))
        
        lines.append(f'{model_name} & {fmt(ad_f1)} & {fmt(ad_p)} & {fmt(ad_r)} '
                     f'& {fmt(sc_f1)} & {fmt(sc_p)} & {fmt(sc_r)} \\\\')
    
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    
    return '\n'.join(lines)


def generate_table2_per_aspect(all_metrics, output_dir):
    """Table 2: Per-aspect F1 comparison for top models."""
    # Get aspect names from first available model
    aspects = []
    for m in all_metrics.values():
        ad_pa = m.get('ad', {}).get('per_aspect', {})
        if ad_pa:
            aspects = list(ad_pa.keys())
            break
        sc_pa = m.get('sc', {}).get('per_aspect', {})
        if sc_pa:
            aspects = list(sc_pa.keys())
            break
    
    if not aspects:
        return '% No per-aspect data available'
    
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{F1 score theo từng aspect cho task Aspect Detection}')
    lines.append(r'\label{tab:per_aspect_ad}')
    
    n_models = len(MODELS_ORDER)
    col_spec = 'l|' + 'c' * n_models
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\hline')
    
    header = r'\textbf{Aspect}'
    for model_name, _ in MODELS_ORDER:
        short = model_name.replace('ViSoBERT-', 'VB-').replace('PhoBERT-', 'PB-').replace('BiLSTM-', 'BL-')
        header += f' & \\textbf{{{short}}}'
    header += r' \\'
    lines.append(header)
    lines.append(r'\hline')
    
    for aspect in aspects:
        row = aspect.replace('_', r'\_')
        for model_name, _ in MODELS_ORDER:
            m = all_metrics.get(model_name, {})
            pa = m.get('ad', {}).get('per_aspect', {})
            if aspect in pa:
                f1 = pa[aspect].get('f1', 0)
                row += f' & {fmt(f1)}'
            else:
                row += ' & ---'
        row += r' \\'
        lines.append(row)
    
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    
    return '\n'.join(lines)


def generate_table3_mcnemar(output_dir):
    """Table 3: McNemar's test results."""
    mcnemar_path = os.path.join(output_dir, 'mcnemar_results.json')
    if not os.path.exists(mcnemar_path):
        return '% McNemar results not available yet'
    
    with open(mcnemar_path) as f:
        results = json.load(f)
    
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r"\caption{Kết quả kiểm định McNemar's test giữa các cặp mô hình}")
    lines.append(r'\label{tab:mcnemar_results}')
    lines.append(r'\begin{tabular}{l|l|r|r|c}')
    lines.append(r'\hline')
    lines.append(r'\textbf{Cặp so sánh} & \textbf{Task} & \textbf{$\chi^2$} & \textbf{p-value} & \textbf{Sig.} \\')
    lines.append(r'\hline')
    
    for r in results:
        pair = f"{r['model_a']} vs {r['model_b']}"
        pair = pair.replace('ViSoBERT', 'VB').replace('PhoBERT', 'PB').replace('BiLSTM', 'BL')
        task = 'AD' if 'Aspect' in r['task'] else 'SC'
        chi2 = f"{r['statistic']:.2f}"
        pval = f"{r['p_value']:.4f}" if r['p_value'] >= 0.0001 else f"{r['p_value']:.2e}"
        sig = r'$\checkmark$' if r.get('bonferroni_significant', r['p_value'] < 0.05) else ''
        lines.append(f'{pair} & {task} & {chi2} & {pval} & {sig} \\\\')
    
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    bonf = results[0].get('bonferroni_alpha', 0.05/len(results)) if results else 0.05
    lines.append(r'\vspace{0.5em}')
    lines.append(f'\\footnotesize{{Bonferroni $\\alpha = {bonf:.4f}$}}')
    lines.append(r'\end{table}')
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables for thesis')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    results_dir = args.results_dir
    output_dir = args.output_dir or os.path.join(results_dir, 'error_analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Generating LaTeX Tables for Thesis")
    print("=" * 80)
    
    all_metrics = load_model_metrics(results_dir)
    
    table1 = generate_table1_overall(all_metrics, output_dir)
    table2 = generate_table2_per_aspect(all_metrics, output_dir)
    table3 = generate_table3_mcnemar(output_dir)
    
    # Combine
    full_tex = '\n\n% ============================================================\n'.join([
        '% Auto-generated by generate_thesis_tables.py',
        '% Table 1: Overall Results\n' + table1,
        '% Table 2: Per-Aspect AD F1\n' + table2,
        "% Table 3: McNemar's Test\n" + table3,
    ])
    
    out_path = os.path.join(output_dir, 'thesis_tables.tex')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(full_tex)
    
    print(f"\nSaved: {out_path}")
    print(f"Tables generated: 3")


if __name__ == '__main__':
    main()

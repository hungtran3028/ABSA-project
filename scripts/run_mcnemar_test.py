import json
import os
import glob
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

def load_predictions(model_name_dir, task):
    """
    Search for `test_predictions_detailed.csv` or similar files in the result directory
    Returns a dataframe of predictions.
    """
    if task == 'ad':
        search_pattern = f"{model_name_dir}/results/**/predictions_detailed_ad.csv"
    else:
        search_pattern = f"{model_name_dir}/results/**/predictions_detailed_sc.csv"
        
    files = glob.glob(search_pattern, recursive=True)
    if not files:
        # Try alternate pattern
        search_pattern = f"{model_name_dir}/results/**/test_results_{task}.csv"
        files = glob.glob(search_pattern, recursive=True)
        
    if files:
        print(f"Found predictions for {model_name_dir} [{task}]: {files[0]}")
        return pd.read_csv(files[0])
    
    print(f"Warning: Predictions not found for {model_name_dir} [{task}]")
    return None


def run_mcnemar(df_model1, df_model2, metric_col='is_correct'):
    """
    Run McNemar's test comparing two models.
    Requires a dataframe with an 'is_correct' boolean/binary column for each sample.
    """
    if df_model1 is None or df_model2 is None:
        return 'N/A'
        
    # Ensure they have identical indices
    assert len(df_model1) == len(df_model2), "Dataframes must have the same length"
    
    b = sum((df_model1[metric_col] == 1) & (df_model2[metric_col] == 0))
    c = sum((df_model1[metric_col] == 0) & (df_model2[metric_col] == 1))
    
    table = [[sum((df_model1[metric_col] == 1) & (df_model2[metric_col] == 1)), b],
             [c, sum((df_model1[metric_col] == 0) & (df_model2[metric_col] == 0))]]
    
    result = mcnemar(table, exact=True)
    return result.pvalue


def main():
    print("Running McNemar's Tests on ABSA Models...")
    
    models = {
        'ViSoBERT-MTL': 'VisoBERT-MTL',
        'ViSoBERT-STL': 'VisoBERT-STL',
        'PhoBERT-MTL': 'phoBERT-MTL',
        'PhoBERT-STL': 'PhoBERT-STL',
        'BiLSTM-MTL': 'BILSTM-MTL',
        'BiLSTM-STL': 'BILSTM-STL'
    }
    
    # Load all AD predictions
    ad_preds = {name: load_predictions(dir, 'ad') for name, dir in models.items()}
    # Load all SC predictions
    sc_preds = {name: load_predictions(dir, 'sc') for name, dir in models.items()}
    
    # Define hypotheses to test
    comparisons = [
        # H1: MTL vs STL
        ('ViSoBERT-MTL', 'ViSoBERT-STL', 'MTL vs STL (ViSoBERT)'),
        ('PhoBERT-MTL', 'PhoBERT-STL', 'MTL vs STL (PhoBERT)'),
        ('BiLSTM-MTL', 'BiLSTM-STL', 'MTL vs STL (BiLSTM)'),
        # H2: ViSoBERT vs PhoBERT
        ('ViSoBERT-MTL', 'PhoBERT-MTL', 'ViSoBERT vs PhoBERT (MTL)'),
        ('ViSoBERT-STL', 'PhoBERT-STL', 'ViSoBERT vs PhoBERT (STL)'),
        # H3: Transformer vs BiLSTM
        ('ViSoBERT-MTL', 'BiLSTM-MTL', 'Transformer vs Deep Learning (MTL)'),
    ]
    
    print("\n" + "="*80)
    print("McNemar's Test Results (p-values)")
    print("="*80)
    print(f"{'Hypothesis':<40} | {'AD':<15} | {'SC':<15}")
    print("-" * 75)
    
    for m1, m2, desc in comparisons:
        # Check if 'is_correct' exists, otherwise standard 'ad_correct' naming maybe used
        # We assume there is a column that tells us if the prediction is correct.
        
        # Check AD
        if ad_preds[m1] is not None and ad_preds[m2] is not None:
            col_name = 'is_correct' if 'is_correct' in ad_preds[m1].columns else 'Exact_Match'
            if col_name in ad_preds[m1].columns:
                p_ad = run_mcnemar(ad_preds[m1], ad_preds[m2], col_name)
                p_ad = f"{p_ad:.4f}" if isinstance(p_ad, float) else p_ad
            else:
                p_ad = 'Col missing'
        else:
            p_ad = 'N/A'
            
        # Check SC
        if sc_preds[m1] is not None and sc_preds[m2] is not None:
            col_name = 'is_correct' if 'is_correct' in sc_preds[m1].columns else 'Exact_Match'
            if col_name in sc_preds[m1].columns:
                p_sc = run_mcnemar(sc_preds[m1], sc_preds[m2], col_name)
                p_sc = f"{p_sc:.4f}" if isinstance(p_sc, float) else p_sc
            else:
                p_sc = 'Col missing'
        else:
            p_sc = 'N/A'
            
        print(f"{desc:<40} | {p_ad:<15} | {p_sc:<15}")

if __name__ == "__main__":
    main()

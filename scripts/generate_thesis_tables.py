import os
import json
import glob

def find_metrics(model_dir, task):
    """
    Search for eval_metrics.json or similar.
    Returns dictionary with f1/acc if found.
    """
    search_pattern = f"{model_dir}/results/**/eval_metrics.json"
    files = glob.glob(search_pattern, recursive=True)
    if not files:
        # Check an alternate pattern
        search_pattern = f"{model_dir}/results/**/test_metrics_{task}.json"
        files = glob.glob(search_pattern, recursive=True)
        
    if not files:
        return None
        
    with open(files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if task in data:
        metrics = data[task]
    else:
        metrics = data
        
    # Standardize keys
    f1 = metrics.get('overall_f1', metrics.get('test_f1', metrics.get('f1_score', 0))) * 100
    acc = metrics.get('overall_accuracy', metrics.get('test_accuracy', metrics.get('accuracy', 0))) * 100
    return {"f1": f1, "acc": acc}

def generate_summary_table():
    models = {
        'ViSoBERT-MTL': 'VisoBERT-MTL',
        'ViSoBERT-STL': 'VisoBERT-STL',
        'PhoBERT-MTL': 'phoBERT-MTL',
        'PhoBERT-STL': 'PhoBERT-STL',
        'BiLSTM-MTL': 'BILSTM-MTL',
        'BiLSTM-STL': 'BILSTM-STL'
    }
    
    results = {}
    for pretty_name, dir_name in models.items():
        ad_metrics = find_metrics(dir_name, 'ad')
        sc_metrics = find_metrics(dir_name, 'sc')
        results[pretty_name] = {
            'ad': ad_metrics if ad_metrics else {'f1': 0, 'acc': 0},
            'sc': sc_metrics if sc_metrics else {'f1': 0, 'acc': 0}
        }
        
    print(r"\begin{table}[h!]")
    print(r"    \centering")
    print(r"    \begin{tabular}{|l|c|c|c|c|c|}")
    print(r"        \hline")
    print(r"        \rowcolor{gray!20}")
    print(r"        \textbf{Model} & \multicolumn{2}{c|}{\textbf{Aspect Detection}} & \multicolumn{2}{c|}{\textbf{Sentiment Classification}} & \textbf{Combined F1} \\ \cline{2-5}")
    print(r"        \rowcolor{gray!20}")
    print(r"        & Accuracy & F1-Score & Accuracy & F1-Score & \\ ")
    print(r"        \hline")
    
    for name, metrics in results.items():
        ad_f1 = metrics['ad']['f1']
        ad_acc = metrics['ad']['acc']
        sc_f1 = metrics['sc']['f1']
        sc_acc = metrics['sc']['acc']
        
        # Combined F1 is often 2*(ad_f1 * sc_f1)/(ad_f1 + sc_f1) or just (ad_f1 + sc_f1)/2
        comb_f1 = (ad_f1 + sc_f1) / 2 if (ad_f1 + sc_f1) > 0 else 0
        
        name_str = name.replace("-", " ")
        print(f"        {name_str} & {ad_acc:.2f}\\% & {ad_f1:.2f}\\% & {sc_acc:.2f}\\% & {sc_f1:.2f}\\% & {comb_f1:.2f}\\% \\\\")
        print(r"        \hline")

    print(r"    \end{tabular}")
    print(r"    \caption{Tổng hợp kết quả đánh giá các mô hình trên tập kiểm thử}")
    print(r"    \label{tab:summary_results}")
    print(r"\end{table}")

if __name__ == "__main__":
    generate_summary_table()

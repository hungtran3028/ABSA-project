"""
BiLSTM Multi-Task Learning for Vietnamese ABSA
==============================================
Train both AD and SC simultaneously with shared backbone

Uses PhoBERT as frozen feature extractor (pre-computed last_hidden_state)
Combined Loss = α * Loss_AD + β * Loss_SC
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import yaml
import argparse
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
import wandb

from model_bilstm_mtl import BiLSTM_MTL
from dataset_bilstm_mtl import MTLEmbeddingDataset
from focal_loss_multilabel import MultilabelFocalLoss, calculate_global_alpha


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: str) -> str:
    """Setup logging"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'mtl_training_log_{timestamp}.txt')
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file


def train_epoch_mtl(model, dataloader, optimizer, scheduler, device, 
                    ad_criterion, sc_focal_loss, loss_weight_ad, loss_weight_sc, scaler):
    """Train one epoch with multi-task learning"""
    model.train()
    total_loss = 0
    total_ad_loss = 0
    total_sc_loss = 0
    
    for batch in tqdm(dataloader, desc="[MTL] Training"):
        embeddings = batch['embeddings'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ad_labels = batch['ad_labels'].to(device)
        sc_labels = batch['sc_labels'].to(device)
        sc_mask = batch['sc_mask'].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            ad_logits, sc_logits = model(embeddings, attention_mask)
            
            # AD loss
            ad_loss = ad_criterion(ad_logits, ad_labels)
            
            # SC loss (masked focal loss)
            sc_loss_per_aspect = sc_focal_loss(sc_logits, sc_labels)  # [bsz, num_aspects]
            sc_masked_loss = sc_loss_per_aspect * sc_mask
            num_labeled = sc_mask.sum()
            sc_loss = sc_masked_loss.sum() / num_labeled if num_labeled > 0 else sc_masked_loss.sum()
            
            loss = loss_weight_ad * ad_loss + loss_weight_sc * sc_loss
        
        if scaler:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        total_ad_loss += ad_loss.item()
        total_sc_loss += sc_loss.item()
    
    return total_loss / len(dataloader), total_ad_loss / len(dataloader), total_sc_loss / len(dataloader)


def evaluate_mtl(model, dataloader, device, aspect_names):
    """Evaluate both AD and SC tasks"""
    model.eval()
    
    ad_preds_all, ad_labels_all = [], []
    sc_preds_all, sc_labels_all, sc_masks_all = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[MTL] Evaluating"):
            embeddings = batch['embeddings'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            ad_logits, sc_logits = model(embeddings, attention_mask)
            
            ad_preds = (torch.sigmoid(ad_logits) >= 0.5).float()
            sc_preds = torch.argmax(sc_logits, dim=-1)
            
            ad_preds_all.append(ad_preds.cpu())
            ad_labels_all.append(batch['ad_labels'])
            sc_preds_all.append(sc_preds.cpu())
            sc_labels_all.append(batch['sc_labels'])
            sc_masks_all.append(batch['sc_mask'])
    
    ad_preds_all = torch.cat(ad_preds_all, dim=0).numpy()
    ad_labels_all = torch.cat(ad_labels_all, dim=0).numpy()
    sc_preds_all = torch.cat(sc_preds_all, dim=0)
    sc_labels_all = torch.cat(sc_labels_all, dim=0)
    sc_masks_all = torch.cat(sc_masks_all, dim=0)
    
    # AD Metrics
    ad_aspect_metrics = {}
    for i, aspect in enumerate(aspect_names):
        acc = accuracy_score(ad_labels_all[:, i], ad_preds_all[:, i])
        p, r, f1, _ = precision_recall_fscore_support(
            ad_labels_all[:, i], ad_preds_all[:, i], average='binary', zero_division=0
        )
        ad_aspect_metrics[aspect] = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}
    
    ad_f1 = np.mean([m['f1'] for m in ad_aspect_metrics.values()])
    
    # SC Metrics (only on labeled aspects)
    sc_aspect_metrics = {}
    for i, aspect in enumerate(aspect_names):
        mask = sc_masks_all[:, i] > 0
        if mask.sum() == 0:
            sc_aspect_metrics[aspect] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
            continue
        aspect_preds = sc_preds_all[:, i][mask].numpy()
        aspect_labels = sc_labels_all[:, i][mask].numpy()
        
        acc = accuracy_score(aspect_labels, aspect_preds)
        p, r, f1, _ = precision_recall_fscore_support(
            aspect_labels, aspect_preds, average='macro', zero_division=0
        )
        sc_aspect_metrics[aspect] = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}
    
    sc_f1 = np.mean([m['f1'] for m in sc_aspect_metrics.values()])
    
    return {
        'ad': {
            'overall_f1': ad_f1,
            'overall_accuracy': np.mean([m['accuracy'] for m in ad_aspect_metrics.values()]),
            'overall_precision': np.mean([m['precision'] for m in ad_aspect_metrics.values()]),
            'overall_recall': np.mean([m['recall'] for m in ad_aspect_metrics.values()]),
            'per_aspect': ad_aspect_metrics,
            'predictions': ad_preds_all,
            'labels': ad_labels_all
        },
        'sc': {
            'overall_f1': sc_f1,
            'overall_accuracy': np.mean([m['accuracy'] for m in sc_aspect_metrics.values()]),
            'overall_precision': np.mean([m['precision'] for m in sc_aspect_metrics.values()]),
            'overall_recall': np.mean([m['recall'] for m in sc_aspect_metrics.values()]),
            'per_aspect': sc_aspect_metrics,
            'predictions': sc_preds_all,
            'labels': sc_labels_all
        }
    }


def save_confusion_matrices(metrics: dict, aspect_names: list, output_dir: str):
    """Save confusion matrices for both AD and SC"""
    print("\n[MTL] Generating confusion matrices...")
    
    # AD confusion matrix
    ad_preds = metrics['ad']['predictions']
    ad_labels = metrics['ad']['labels']
    cm_ad = confusion_matrix(ad_labels.flatten(), ad_preds.flatten())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_ad, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Not Mentioned', 'Mentioned'],
               yticklabels=['Not Mentioned', 'Mentioned'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Aspect Detection - Confusion Matrix (BiLSTM-MTL)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_ad.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # SC overall confusion matrix
    sc_preds = metrics['sc']['predictions'].numpy()
    sc_labels = metrics['sc']['labels'].numpy()
    cm_sc = confusion_matrix(sc_labels.flatten(), sc_preds.flatten(), labels=[0, 1, 2])
    
    sentiment_labels = ['Positive', 'Negative', 'Neutral']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_sc, annot=True, fmt='d', cmap='Blues',
               xticklabels=sentiment_labels, yticklabels=sentiment_labels, ax=ax)
    ax.set_xlabel('Predicted Sentiment')
    ax.set_ylabel('True Sentiment')
    ax.set_title('Sentiment Classification - Overall (BiLSTM-MTL)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_sc_overall.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {output_dir}/confusion_matrix_ad.png")
    print(f"   Saved: {output_dir}/confusion_matrix_sc_overall.png")


def generate_final_report(metrics: dict, output_dir: str, config: dict):
    """Generate final report"""
    report_lines = [
        "=" * 80,
        "BILSTM MULTI-TASK LEARNING FOR VIETNAMESE ABSA",
        "=" * 80, "",
        f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: BiLSTM + PhoBERT Feature Extractor + Additive Attention",
        f"Loss Weights: AD={config['multi_task']['loss_weight_ad']}, SC={config['multi_task']['loss_weight_sc']}",
        "",
        "=" * 80,
        "TASK 1: ASPECT DETECTION (AD)", "=" * 80, "",
        f"Test F1 Score:  {metrics['ad']['overall_f1']*100:.2f}%",
        "",
        "=" * 80,
        "TASK 2: SENTIMENT CLASSIFICATION (SC)", "=" * 80, "",
        f"Test F1 Score:  {metrics['sc']['overall_f1']*100:.2f}%",
    ]
    
    report_text = '\n'.join(report_lines)
    
    with open(os.path.join(output_dir, 'final_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)


def main(args: argparse.Namespace):
    """Main training function"""
    print("=" * 80)
    print("BILSTM MULTI-TASK LEARNING FOR VIETNAMESE ABSA")
    print("Using PhoBERT as Feature Extractor")
    print("=" * 80)
    
    # Load config
    config = load_config(args.config)
    output_dir = config['paths']['output_dir']
    log_file = setup_logging(output_dir)
    
    # Initialize wandb
    try:
        wandb_key = os.environ.get("WANDB_API_KEY")
        if wandb_key:
            wandb.login(key=wandb_key)
        else:
            wandb.login()
        wandb.init(project="ABSA-Vietnamese", name="BiLSTM-MTL", config=config, tags=["mtl", "bilstm"])
    except Exception as e:
        logging.warning(f"Wandb init failed: {e}")
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(mode="disabled")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Set seed
    seed = config['reproducibility']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load PhoBERT tokenizer + model (feature extractor)
    print("\nLoading PhoBERT tokenizer + model...")
    embedding_model_name = config['model']['embedding_model']
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    embedding_model = AutoModel.from_pretrained(embedding_model_name).to(device)
    embedding_model.eval()
    for param in embedding_model.parameters():
        param.requires_grad = False
    print(f"   PhoBERT feature extractor loaded: {embedding_model_name} (FROZEN)")
    
    # Datasets (pre-compute PhoBERT embeddings)
    print("\nLoading datasets (pre-computing embeddings)...")
    max_length = config['model']['max_length']
    
    train_dataset = MTLEmbeddingDataset(
        config['paths']['train_file'], tokenizer, embedding_model, max_length, device
    )
    val_dataset = MTLEmbeddingDataset(
        config['paths']['validation_file'], tokenizer, embedding_model, max_length, device
    )
    test_dataset = MTLEmbeddingDataset(
        config['paths']['test_file'], tokenizer, embedding_model, max_length, device
    )
    
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Free embedding model from GPU
    del embedding_model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("   PhoBERT model freed from GPU ✅")
    
    # Dataloaders
    batch_size = config['training']['per_device_train_batch_size']
    eval_batch_size = config['training']['per_device_eval_batch_size']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    
    # BiLSTM Model
    print("\nCreating BiLSTM MTL model...")
    model = BiLSTM_MTL(
        embedding_dim=config['model']['embedding_dim'],
        num_aspects=config['model']['num_aspects'],
        num_sentiments=config['model']['num_sentiments'],
        lstm_hidden_size=config['model']['lstm_hidden_size'],
        lstm_num_layers=config['model']['lstm_num_layers'],
        lstm_dropout=config['model']['lstm_dropout'],
        classifier_dropout=config['model']['classifier_dropout'],
        bidirectional=config['model']['bidirectional']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Loss functions
    mtl_config = config['multi_task']
    ad_criterion = torch.nn.BCEWithLogitsLoss()
    loss_weight_ad = mtl_config['loss_weight_ad']
    loss_weight_sc = mtl_config['loss_weight_sc']
    print(f"   Loss weights: AD={loss_weight_ad}, SC={loss_weight_sc}")
    
    # SC Focal Loss (standardized across all models)
    sc_config = mtl_config.get('sentiment_classification', {})
    focal_gamma = sc_config.get('focal_gamma', 2)
    focal_alpha_setting = sc_config.get('focal_alpha', 'auto')
    
    if focal_alpha_setting == 'auto':
        aspect_cols = config['aspect_names']
        sentiment_to_idx = config['sentiment_labels']
        focal_alpha = calculate_global_alpha(
            config['paths']['train_file'], aspect_cols, sentiment_to_idx
        )
    else:
        focal_alpha = focal_alpha_setting
    
    sc_focal_loss = MultilabelFocalLoss(
        alpha=focal_alpha, gamma=focal_gamma, 
        num_aspects=config['model']['num_aspects'], reduction='none'
    ).to(device)
    print(f"   SC Loss: MultilabelFocalLoss(gamma={focal_gamma}, alpha={focal_alpha})")
    
    # Optimizer & Scheduler
    num_epochs = config['training']['num_train_epochs']
    learning_rate = config['training']['learning_rate']
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=config['training']['weight_decay'])
    
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(config['training']['warmup_ratio'] * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    print(f"\nTraining: {num_epochs} epochs, batch {batch_size}, lr {learning_rate}")
    
    # Training loop
    best_combined_f1 = 0.0
    history = []
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    patience_counter = 0
    early_stopping_patience = config['training']['early_stopping_patience']
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}\n[MTL] Epoch {epoch}/{num_epochs}\n{'='*60}")
        
        train_loss, train_ad_loss, train_sc_loss = train_epoch_mtl(
            model, train_loader, optimizer, scheduler, device,
            ad_criterion, sc_focal_loss, loss_weight_ad, loss_weight_sc, scaler
        )
        print(f"Train Loss: {train_loss:.4f} (AD: {train_ad_loss:.4f}, SC: {train_sc_loss:.4f})")
        
        val_metrics = evaluate_mtl(model, val_loader, device, train_dataset.aspects)
        val_ad_f1 = val_metrics['ad']['overall_f1']
        val_sc_f1 = val_metrics['sc']['overall_f1']
        combined_f1 = (val_ad_f1 + val_sc_f1) / 2
        
        print(f"Val AD F1: {val_ad_f1*100:.2f}%, SC F1: {val_sc_f1*100:.2f}%, Combined: {combined_f1*100:.2f}%")
        
        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_ad_f1': val_ad_f1, 'val_sc_f1': val_sc_f1, 'val_combined_f1': combined_f1
        })
        
        wandb.log({
            'epoch': epoch, 'train/loss': train_loss,
            'val/ad_f1': val_ad_f1, 'val/sc_f1': val_sc_f1, 'val/combined_f1': combined_f1
        })
        
        if combined_f1 > best_combined_f1:
            best_combined_f1 = combined_f1
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': val_metrics
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"✓ New best! Combined F1: {best_combined_f1*100:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch} epochs")
                break
    
    pd.DataFrame(history).to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # Test
    print(f"\n{'='*60}\n[MTL] Testing Best Model\n{'='*60}")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_mtl(model, test_loader, device, train_dataset.aspects)
    print(f"Test AD F1: {test_metrics['ad']['overall_f1']*100:.2f}%")
    print(f"Test SC F1: {test_metrics['sc']['overall_f1']*100:.2f}%")
    
    # Save results
    results = {
        'ad': {
            'test_f1': test_metrics['ad']['overall_f1'],
            'test_precision': test_metrics['ad']['overall_precision'],
            'test_recall': test_metrics['ad']['overall_recall'],
            'per_aspect': test_metrics['ad']['per_aspect']
        },
        'sc': {
            'test_f1': test_metrics['sc']['overall_f1'],
            'test_precision': test_metrics['sc']['overall_precision'],
            'test_recall': test_metrics['sc']['overall_recall'],
            'per_aspect': test_metrics['sc']['per_aspect']
        },
        'training_completed': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    save_confusion_matrices(test_metrics, train_dataset.aspects, output_dir)
    generate_final_report(test_metrics, output_dir, config)
    
    print(f"\n✅ BiLSTM MTL training complete! Results: {output_dir}")
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BiLSTM Multi-Task Learning for ABSA')
    parser.add_argument('--config', type=str, default='BILSTM-MTL/config_bilstm_mtl.yaml')
    args = parser.parse_args()
    main(args)

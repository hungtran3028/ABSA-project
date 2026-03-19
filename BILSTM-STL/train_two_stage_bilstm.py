"""
Sequential Single-Task Learning for Vietnamese ABSA using BiLSTM
================================================================
Two-stage training with PhoBERT as feature extractor:
    Stage 1: Aspect Detection (AD)
    Stage 2: Sentiment Classification (SC)
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

from model_bilstm_ad import BiLSTM_AD
from dataset_bilstm_ad import ADEmbeddingDataset
from model_bilstm_sc import BiLSTM_SC
from dataset_bilstm_sc import SCEmbeddingDataset


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: str, stage_name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'{stage_name}_log_{timestamp}.txt')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
    )
    return log_file


# =============================================================================
# STAGE 1: ASPECT DETECTION
# =============================================================================

def train_epoch_ad(model, dataloader, optimizer, scheduler, device, criterion, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="[AD] Training"):
        embeddings = batch['embeddings'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['ad_labels'].to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(embeddings, attention_mask)
            loss = criterion(logits, labels)
        
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
    return total_loss / len(dataloader)


def evaluate_ad(model, dataloader, device, aspect_names):
    model.eval()
    preds_all, labels_all = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[AD] Evaluating"):
            embeddings = batch['embeddings'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(embeddings, attention_mask)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            preds_all.append(preds.cpu())
            labels_all.append(batch['ad_labels'])
    
    preds_all = torch.cat(preds_all, dim=0).numpy()
    labels_all = torch.cat(labels_all, dim=0).numpy()
    
    aspect_metrics = {}
    for i, aspect in enumerate(aspect_names):
        acc = accuracy_score(labels_all[:, i], preds_all[:, i])
        p, r, f1, _ = precision_recall_fscore_support(
            labels_all[:, i], preds_all[:, i], average='binary', zero_division=0
        )
        aspect_metrics[aspect] = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}
    
    overall_f1 = np.mean([m['f1'] for m in aspect_metrics.values()])
    
    return {
        'overall_f1': overall_f1,
        'overall_accuracy': np.mean([m['accuracy'] for m in aspect_metrics.values()]),
        'per_aspect': aspect_metrics,
        'predictions': preds_all,
        'labels': labels_all
    }


def train_aspect_detection(config, embedding_model, tokenizer, device):
    """Train Stage 1: Aspect Detection"""
    print("\n" + "=" * 80)
    print("STAGE 1: ASPECT DETECTION (BiLSTM + PhoBERT Features)")
    print("=" * 80)
    
    output_dir = config['paths']['ad_output_dir']
    setup_logging(output_dir, 'aspect_detection')
    
    seed = config['reproducibility']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    max_length = config['model']['max_length']
    
    print("\nLoading datasets (pre-computing AD embeddings)...")
    train_dataset = ADEmbeddingDataset(config['paths']['train_file'], tokenizer, embedding_model, max_length, device)
    val_dataset = ADEmbeddingDataset(config['paths']['validation_file'], tokenizer, embedding_model, max_length, device)
    test_dataset = ADEmbeddingDataset(config['paths']['test_file'], tokenizer, embedding_model, max_length, device)
    
    aspects = train_dataset.aspects
    batch_size = config['training']['per_device_train_batch_size']
    eval_batch_size = config['training']['per_device_eval_batch_size']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    
    model = BiLSTM_AD(
        embedding_dim=config['model']['embedding_dim'],
        hidden_size=config['model']['lstm_hidden_size'],
        num_layers=config['model']['lstm_num_layers'],
        num_aspects=config['model']['num_aspects'],
        dropout=config['model']['lstm_dropout'],
        bidirectional=config['model']['bidirectional']
    ).to(device)
    
    print(f"BiLSTM_AD params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = torch.nn.BCEWithLogitsLoss()
    ad_config = config['two_stage']['aspect_detection']
    num_epochs = ad_config.get('epochs', 20)
    learning_rate = config['training']['learning_rate']
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=config['training']['weight_decay'])
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(config['training']['warmup_ratio'] * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_f1 = 0.0
    history = []
    patience_counter = 0
    patience = ad_config.get('early_stopping_patience', 5)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n[AD] Epoch {epoch}/{num_epochs}")
        
        train_loss = train_epoch_ad(model, train_loader, optimizer, scheduler, device, criterion, scaler)
        val_metrics = evaluate_ad(model, val_loader, device, aspects)
        val_f1 = val_metrics['overall_f1']
        
        print(f"Loss: {train_loss:.4f}, Val F1: {val_f1*100:.2f}%")
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_f1': val_f1})
        wandb.log({'ad/train_loss': train_loss, 'ad/val_f1': val_f1, 'ad/epoch': epoch})
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                      os.path.join(output_dir, 'best_model.pt'))
            print(f"✓ New best AD F1: {best_f1*100:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping AD after {epoch} epochs")
                break
    
    pd.DataFrame(history).to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # Test
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate_ad(model, test_loader, device, aspects)
    print(f"\n[AD] Test F1: {test_metrics['overall_f1']*100:.2f}%")
    
    results = {'test_f1': test_metrics['overall_f1'], 'per_aspect': test_metrics['per_aspect'],
               'training_completed': datetime.now().isoformat()}
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Confusion matrix
    cm = confusion_matrix(test_metrics['labels'].flatten(), test_metrics['predictions'].flatten())
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Not Mentioned', 'Mentioned'],
               yticklabels=['Not Mentioned', 'Mentioned'], ax=ax)
    ax.set_title('AD Confusion Matrix (BiLSTM-STL)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_ad.png'), dpi=300)
    plt.close()
    
    return output_dir


# =============================================================================
# STAGE 2: SENTIMENT CLASSIFICATION
# =============================================================================

def train_epoch_sc(model, dataloader, optimizer, scheduler, device, criterion, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="[SC] Training"):
        embeddings = batch['embeddings'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sc_labels = batch['sc_labels'].to(device)
        sc_mask = batch['sc_mask'].to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            sc_logits = model(embeddings, attention_mask)
            bsz, num_aspects, num_classes = sc_logits.shape
            ce = criterion(sc_logits.view(bsz * num_aspects, num_classes), sc_labels.view(bsz * num_aspects))
            sc_loss_per_aspect = ce.view(bsz, num_aspects)
            masked = sc_loss_per_aspect * sc_mask
            num_labeled = sc_mask.sum()
            loss = masked.sum() / num_labeled if num_labeled > 0 else masked.sum()
        
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
    return total_loss / len(dataloader)


def evaluate_sc(model, dataloader, device, aspect_names):
    model.eval()
    preds_all, labels_all, masks_all = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[SC] Evaluating"):
            embeddings = batch['embeddings'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sc_logits = model(embeddings, attention_mask)
            sc_preds = torch.argmax(sc_logits, dim=-1)
            preds_all.append(sc_preds.cpu())
            labels_all.append(batch['sc_labels'])
            masks_all.append(batch['sc_mask'])
    
    preds_all = torch.cat(preds_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    masks_all = torch.cat(masks_all, dim=0)
    
    aspect_metrics = {}
    for i, aspect in enumerate(aspect_names):
        mask = masks_all[:, i] > 0
        if mask.sum() == 0:
            aspect_metrics[aspect] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
            continue
        p_aspect = preds_all[:, i][mask].numpy()
        l_aspect = labels_all[:, i][mask].numpy()
        acc = accuracy_score(l_aspect, p_aspect)
        p, r, f1, _ = precision_recall_fscore_support(l_aspect, p_aspect, average='macro', zero_division=0)
        aspect_metrics[aspect] = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}
    
    overall_f1 = np.mean([m['f1'] for m in aspect_metrics.values()])
    
    return {
        'overall_f1': overall_f1,
        'overall_accuracy': np.mean([m['accuracy'] for m in aspect_metrics.values()]),
        'per_aspect': aspect_metrics,
        'predictions': preds_all,
        'labels': labels_all
    }


def train_sentiment_classification(config, embedding_model, tokenizer, device):
    """Train Stage 2: Sentiment Classification"""
    print("\n" + "=" * 80)
    print("STAGE 2: SENTIMENT CLASSIFICATION (BiLSTM + PhoBERT Features)")
    print("=" * 80)
    
    output_dir = config['paths']['sc_output_dir']
    setup_logging(output_dir, 'sentiment_classification')
    
    seed = config['reproducibility']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    max_length = config['model']['max_length']
    
    print("\nLoading datasets (pre-computing SC embeddings)...")
    train_dataset = SCEmbeddingDataset(config['paths']['train_file'], tokenizer, embedding_model, max_length, device)
    val_dataset = SCEmbeddingDataset(config['paths']['validation_file'], tokenizer, embedding_model, max_length, device)
    test_dataset = SCEmbeddingDataset(config['paths']['test_file'], tokenizer, embedding_model, max_length, device)
    
    aspects = train_dataset.aspects
    batch_size = config['training']['per_device_train_batch_size']
    eval_batch_size = config['training']['per_device_eval_batch_size']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    
    model = BiLSTM_SC(
        embedding_dim=config['model']['embedding_dim'],
        hidden_size=config['model']['lstm_hidden_size'],
        num_layers=config['model']['lstm_num_layers'],
        num_aspects=config['model']['num_aspects'],
        num_sentiments=config['model']['num_sentiments'],
        dropout=config['model']['lstm_dropout'],
        bidirectional=config['model']['bidirectional']
    ).to(device)
    
    print(f"BiLSTM_SC params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    sc_config = config['two_stage']['sentiment_classification']
    num_epochs = sc_config.get('epochs', 20)
    learning_rate = config['training']['learning_rate']
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=config['training']['weight_decay'])
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(config['training']['warmup_ratio'] * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_f1 = 0.0
    history = []
    patience_counter = 0
    patience = sc_config.get('early_stopping_patience', 5)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n[SC] Epoch {epoch}/{num_epochs}")
        
        train_loss = train_epoch_sc(model, train_loader, optimizer, scheduler, device, criterion, scaler)
        val_metrics = evaluate_sc(model, val_loader, device, aspects)
        val_f1 = val_metrics['overall_f1']
        
        print(f"Loss: {train_loss:.4f}, Val F1: {val_f1*100:.2f}%")
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_f1': val_f1})
        wandb.log({'sc/train_loss': train_loss, 'sc/val_f1': val_f1, 'sc/epoch': epoch})
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                      os.path.join(output_dir, 'best_model.pt'))
            print(f"✓ New best SC F1: {best_f1*100:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping SC after {epoch} epochs")
                break
    
    pd.DataFrame(history).to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # Test
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate_sc(model, test_loader, device, aspects)
    print(f"\n[SC] Test F1: {test_metrics['overall_f1']*100:.2f}%")
    
    results = {'test_f1': test_metrics['overall_f1'], 'per_aspect': test_metrics['per_aspect'],
               'training_completed': datetime.now().isoformat()}
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Confusion matrix
    sc_preds = test_metrics['predictions'].numpy()
    sc_labels = test_metrics['labels'].numpy()
    cm = confusion_matrix(sc_labels.flatten(), sc_preds.flatten(), labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Pos', 'Neg', 'Neu'], yticklabels=['Pos', 'Neg', 'Neu'], ax=ax)
    ax.set_title('SC Confusion Matrix (BiLSTM-STL)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_sc.png'), dpi=300)
    plt.close()
    
    return output_dir


# =============================================================================
# MAIN
# =============================================================================

def main(args: argparse.Namespace):
    """Main function for two-stage BiLSTM training"""
    print("=" * 80)
    print("SEQUENTIAL SINGLE-TASK LEARNING FOR VIETNAMESE ABSA (BiLSTM)")
    print("Using PhoBERT as Feature Extractor")
    print("=" * 80)
    
    config = load_config(args.config)
    
    # Initialize wandb
    try:
        wandb_key = os.environ.get("WANDB_API_KEY")
        if wandb_key:
            wandb.login(key=wandb_key)
        else:
            wandb.login()
        wandb.init(project="ABSA-Vietnamese", name="BiLSTM-STL", config=config, tags=["stl", "bilstm"])
    except Exception as e:
        logging.warning(f"Wandb init failed: {e}")
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(mode="disabled")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load PhoBERT feature extractor (shared for both stages)
    print("\nLoading PhoBERT feature extractor...")
    embedding_model_name = config['model']['embedding_model']
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    embedding_model = AutoModel.from_pretrained(embedding_model_name).to(device)
    embedding_model.eval()
    for param in embedding_model.parameters():
        param.requires_grad = False
    print(f"   PhoBERT loaded: {embedding_model_name} (FROZEN)")
    
    # Stage 1: Aspect Detection
    ad_output_dir = train_aspect_detection(config, embedding_model, tokenizer, device)
    
    # Stage 2: Sentiment Classification
    sc_output_dir = train_sentiment_classification(config, embedding_model, tokenizer, device)
    
    # Free PhoBERT
    del embedding_model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("TWO-STAGE BiLSTM TRAINING COMPLETE!")
    print("=" * 80)
    print(f"  AD: {ad_output_dir}")
    print(f"  SC: {sc_output_dir}")
    
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-Stage BiLSTM ABSA Training')
    parser.add_argument('--config', type=str, default='BILSTM-STL/config_bilstm_stl.yaml')
    args = parser.parse_args()
    main(args)

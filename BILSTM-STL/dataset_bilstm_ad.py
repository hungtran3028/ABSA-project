"""
Dataset for BiLSTM AD (STL) - Pre-computes PhoBERT Embeddings
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

ASPECTS = ['Battery', 'Camera', 'Performance', 'Display', 'Design',
           'Packaging', 'Price', 'Shop_Service', 'Shipping', 'General', 'Others']


class ADEmbeddingDataset(Dataset):
    """Pre-computes PhoBERT embeddings + AD labels"""
    
    def __init__(self, csv_file, tokenizer, embedding_model, max_length, device, aspects=ASPECTS):
        self.df = pd.read_csv(csv_file)
        self.aspects = aspects
        self.max_length = max_length
        
        print(f"Pre-computing PhoBERT embeddings for AD ({len(self.df)} samples)...")
        self.embeddings = []
        self.attention_masks = []
        self.ad_labels = []
        
        embedding_model.eval()
        with torch.no_grad():
            for idx in tqdm(range(len(self.df)), desc="AD embeddings"):
                row = self.df.iloc[idx]
                text = str(row['data'])
                
                encoding = tokenizer(text, max_length=max_length, padding='max_length',
                                   truncation=True, return_tensors='pt')
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                outputs = embedding_model(input_ids=input_ids, attention_mask=attention_mask)
                embedding = outputs.last_hidden_state.squeeze(0).cpu()
                
                self.embeddings.append(embedding)
                self.attention_masks.append(attention_mask.squeeze(0).cpu())
                
                ad_label = torch.zeros(len(aspects))
                for i, aspect in enumerate(aspects):
                    val = row.get(aspect, 0)
                    if pd.notna(val) and val != 0:
                        ad_label[i] = 1
                self.ad_labels.append(ad_label)
        
        print(f"✅ AD embeddings computed")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'embeddings': self.embeddings[idx],
            'attention_mask': self.attention_masks[idx],
            'ad_labels': self.ad_labels[idx]
        }

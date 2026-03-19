"""
Dataset for BiLSTM SC (STL) - Pre-computes PhoBERT Embeddings
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

ASPECTS = ['Battery', 'Camera', 'Performance', 'Display', 'Design',
           'Packaging', 'Price', 'Shop_Service', 'Shipping', 'General', 'Others']
SENTIMENTS = ['positive', 'negative', 'neutral']


class SCEmbeddingDataset(Dataset):
    """Pre-computes PhoBERT embeddings + SC labels"""
    
    def __init__(self, csv_file, tokenizer, embedding_model, max_length, device,
                 aspects=ASPECTS, sentiments=SENTIMENTS):
        self.df = pd.read_csv(csv_file)
        self.aspects = aspects
        self.sentiment_map = {s: i for i, s in enumerate(sentiments)}
        self.max_length = max_length
        
        print(f"Pre-computing PhoBERT embeddings for SC ({len(self.df)} samples)...")
        self.embeddings = []
        self.attention_masks = []
        self.sc_labels = []
        self.sc_masks = []
        
        embedding_model.eval()
        with torch.no_grad():
            for idx in tqdm(range(len(self.df)), desc="SC embeddings"):
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
                
                sc_label = torch.zeros(len(aspects), dtype=torch.long)
                sc_mask = torch.zeros(len(aspects))
                
                for i, aspect in enumerate(aspects):
                    val = row.get(aspect, 0)
                    if pd.notna(val) and val != 0:
                        sc_mask[i] = 1
                        if isinstance(val, str):
                            sc_label[i] = self.sentiment_map.get(val.lower(), 2)
                        else:
                            sc_label[i] = int(val) if val in [0, 1, 2] else 2
                
                self.sc_labels.append(sc_label)
                self.sc_masks.append(sc_mask)
        
        print(f"✅ SC embeddings computed")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'embeddings': self.embeddings[idx],
            'attention_mask': self.attention_masks[idx],
            'sc_labels': self.sc_labels[idx],
            'sc_mask': self.sc_masks[idx]
        }

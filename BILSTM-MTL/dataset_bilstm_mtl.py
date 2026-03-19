"""
Dataset for BiLSTM MTL - Pre-computes PhoBERT Embeddings
========================================================
Pre-computes PhoBERT last_hidden_state for all samples,
then serves embeddings (not input_ids) to BiLSTM.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

ASPECTS = ['Battery', 'Camera', 'Performance', 'Display', 'Design',
           'Packaging', 'Price', 'Shop_Service', 'Shipping', 'General', 'Others']
SENTIMENTS = ['positive', 'negative', 'neutral']


class MTLEmbeddingDataset(Dataset):
    """
    Pre-computes PhoBERT contextualized embeddings for BiLSTM.
    
    Unlike the Transformer models which fine-tune BERT, BiLSTM uses
    frozen PhoBERT as a feature extractor and trains only the BiLSTM + heads.
    """
    
    def __init__(self, csv_file, tokenizer, embedding_model, max_length, device,
                 aspects=ASPECTS, sentiments=SENTIMENTS):
        """
        Args:
            csv_file: Path to CSV file
            tokenizer: PhoBERT tokenizer
            embedding_model: PhoBERT model (frozen, for feature extraction)
            max_length: Maximum sequence length
            device: torch device
        """
        self.df = pd.read_csv(csv_file)
        self.aspects = aspects
        self.sentiment_map = {s: i for i, s in enumerate(sentiments)}
        self.max_length = max_length
        
        # Pre-compute PhoBERT embeddings
        print(f"Pre-computing PhoBERT embeddings for {len(self.df)} samples...")
        self.embeddings = []
        self.attention_masks = []
        self.ad_labels = []
        self.sc_labels = []
        self.sc_masks = []
        
        embedding_model.eval()
        with torch.no_grad():
            for idx in tqdm(range(len(self.df)), desc="Computing embeddings"):
                row = self.df.iloc[idx]
                text = str(row['data'])
                
                # Tokenize
                encoding = tokenizer(
                    text,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Get PhoBERT contextualized embeddings
                outputs = embedding_model(input_ids=input_ids, attention_mask=attention_mask)
                embedding = outputs.last_hidden_state.squeeze(0).cpu()
                
                self.embeddings.append(embedding)
                self.attention_masks.append(attention_mask.squeeze(0).cpu())
                
                # Parse labels
                ad_label = torch.zeros(len(self.aspects))
                sc_label = torch.zeros(len(self.aspects), dtype=torch.long)
                sc_mask = torch.zeros(len(self.aspects))
                
                for i, aspect in enumerate(self.aspects):
                    val = row.get(aspect, 0)
                    if pd.notna(val) and val != 0:
                        ad_label[i] = 1
                        sc_mask[i] = 1
                        if isinstance(val, str):
                            sc_label[i] = self.sentiment_map.get(val.lower(), 2)
                        else:
                            sc_label[i] = int(val) if val in [0, 1, 2] else 2
                
                self.ad_labels.append(ad_label)
                self.sc_labels.append(sc_label)
                self.sc_masks.append(sc_mask)
        
        print(f"✅ Embeddings computed for {len(self.df)} samples")
        self._print_statistics()
    
    def _print_statistics(self):
        """Print dataset statistics"""
        ad_labels = torch.stack(self.ad_labels)
        sc_labels = torch.stack(self.sc_labels)
        sc_masks = torch.stack(self.sc_masks)
        
        print(f"\nDataset Statistics:")
        print(f"  Samples: {len(self.df)}")
        print(f"  Embedding dim: {self.embeddings[0].shape[-1]}")
        print(f"  Max length: {self.max_length}")
        
        # Per-aspect stats
        for i, aspect in enumerate(self.aspects):
            n_pos = int(ad_labels[:, i].sum())
            n_total = len(self.df)
            print(f"  {aspect}: {n_pos}/{n_total} ({100*n_pos/n_total:.1f}%)")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'embeddings': self.embeddings[idx],
            'attention_mask': self.attention_masks[idx],
            'ad_labels': self.ad_labels[idx],
            'sc_labels': self.sc_labels[idx],
            'sc_mask': self.sc_masks[idx]
        }


def test_dataset():
    """Test MTLEmbeddingDataset"""
    print("Testing MTLEmbeddingDataset requires PhoBERT model - skipped in unit test")
    print("This dataset pre-computes PhoBERT last_hidden_state for BiLSTM input")


if __name__ == '__main__':
    test_dataset()

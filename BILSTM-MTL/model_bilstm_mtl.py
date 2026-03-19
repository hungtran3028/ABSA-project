"""
BiLSTM Multi-Task Learning Model for Vietnamese ABSA
=====================================================
Uses PhoBERT as feature extractor (contextualized embeddings)
+ BiLSTM + Additive Attention + LayerNorm

Architecture:
    Input (pre-computed PhoBERT embeddings) → BiLSTM → Additive Attention → Shared Dense
                                    ↓
                        ┌───────────┴───────────┐
                        │                       │
                    AD Head                  SC Head
                Linear → 11              Linear → 11×3
                (Sigmoid)                (Reshape → Softmax)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_MTL(nn.Module):
    """
    BiLSTM Multi-Task Learning Model
    
    Receives pre-computed PhoBERT embeddings (contextualized) as input.
    Shared BiLSTM backbone + Additive Attention + 2 task-specific heads.
    """
    
    def __init__(
        self,
        embedding_dim=768,
        num_aspects=11,
        num_sentiments=3,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        lstm_dropout=0.3,
        classifier_dropout=0.3,
        bidirectional=True
    ):
        super().__init__()
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        
        # Additive Attention
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout)
        )
        
        # Task-specific heads
        self.ad_head = nn.Linear(256, num_aspects)
        self.sc_head = nn.Linear(256, num_aspects * num_sentiments)
    
    def forward(self, embeddings, attention_mask):
        """
        Forward pass
        
        Args:
            embeddings: [batch_size, seq_len, embedding_dim] - pre-computed PhoBERT embeddings
            attention_mask: [batch_size, seq_len]
        
        Returns:
            ad_logits: [batch_size, num_aspects]
            sc_logits: [batch_size, num_aspects, num_sentiments]
        """
        # BiLSTM
        lstm_out, _ = self.lstm(embeddings)  # [batch, seq_len, lstm_hidden*2]
        
        # Additive Attention
        attn_weights = self.attention(lstm_out).squeeze(-1)  # [batch, seq_len]
        attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)  # [batch, seq_len, 1]
        context = (lstm_out * attn_weights).sum(dim=1)  # [batch, lstm_hidden*2]
        
        # Shared layer
        shared = self.shared(context)  # [batch, 256]
        
        # Task heads
        ad_logits = self.ad_head(shared)  # [batch, num_aspects]
        sc_logits = self.sc_head(shared).view(-1, self.num_aspects, self.num_sentiments)
        
        return ad_logits, sc_logits


def test_model():
    """Test BiLSTM MTL model"""
    print("=" * 80)
    print("Testing BiLSTM Multi-Task Learning Model (PhoBERT Feature Extractor)")
    print("=" * 80)
    
    model = BiLSTM_MTL(
        embedding_dim=768,
        num_aspects=11,
        num_sentiments=3,
        lstm_hidden_size=256,
        lstm_num_layers=2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 256
    
    embeddings = torch.randn(batch_size, seq_len, 768)
    attention_mask = torch.ones(batch_size, seq_len)
    
    model.eval()
    with torch.no_grad():
        ad_logits, sc_logits = model(embeddings, attention_mask)
    
    print(f"   Input shape: {embeddings.shape}")
    print(f"   AD logits shape: {ad_logits.shape}  (expected: [{batch_size}, 11])")
    print(f"   SC logits shape: {sc_logits.shape}  (expected: [{batch_size}, 11, 3])")
    
    assert ad_logits.shape == (batch_size, 11), f"AD shape mismatch: {ad_logits.shape}"
    assert sc_logits.shape == (batch_size, 11, 3), f"SC shape mismatch: {sc_logits.shape}"
    
    print("\nAll tests passed! ✅")
    print("=" * 80)


if __name__ == '__main__':
    test_model()

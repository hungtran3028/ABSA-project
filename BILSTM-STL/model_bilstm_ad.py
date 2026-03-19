"""
BiLSTM Aspect Detection Model (STL)
====================================
Receives pre-computed PhoBERT embeddings → BiLSTM → Additive Attention → AD classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_AD(nn.Module):
    """BiLSTM for Aspect Detection (binary classification per aspect)"""
    
    def __init__(self, embedding_dim=768, hidden_size=256, num_layers=2,
                 num_aspects=11, dropout=0.3, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Additive Attention
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_aspects)
        )
    
    def forward(self, embeddings, attention_mask):
        """
        Args:
            embeddings: [batch, seq_len, embedding_dim] - pre-computed PhoBERT embeddings
            attention_mask: [batch, seq_len]
        Returns:
            logits: [batch, num_aspects]
        """
        lstm_out, _ = self.lstm(embeddings)
        
        attn_weights = self.attention(lstm_out).squeeze(-1)
        attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)
        context = (lstm_out * attn_weights).sum(dim=1)
        
        return self.classifier(context)


def test_model():
    model = BiLSTM_AD(embedding_dim=768, num_aspects=11)
    print(f"BiLSTM_AD params: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(4, 256, 768)
    mask = torch.ones(4, 256)
    
    model.eval()
    with torch.no_grad():
        out = model(x, mask)
    
    assert out.shape == (4, 11), f"Shape mismatch: {out.shape}"
    print(f"Output shape: {out.shape} ✅")


if __name__ == '__main__':
    test_model()

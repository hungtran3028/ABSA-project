"""
BiLSTM Multi-Task Learning Model for Vietnamese ABSA
=====================================================
One shared backbone with two task-specific heads:
    - Head 1: Aspect Detection (AD) - Binary classification (11 aspects)
    - Head 2: Sentiment Classification (SC) - 3-class per aspect (11 × 3)

Architecture:
    Input → Embedding → SpatialDropout → BiLSTM → MultiHeadAttention → Conv1D → Pooling
                                ↓
                    ┌───────────┴───────────┐
                    │                       │
                AD Head                  SC Head
            Dense → Sigmoid         Dense → Reshape [11, 3]
            11 binary outputs       11 × 3 multi-class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalPooling(nn.Module):
    """Global Average Pooling + Global Max Pooling"""
    def __init__(self):
        super(GlobalPooling, self).__init__()
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, features]
        Returns:
            pooled: [batch_size, features * 2]
        """
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return pooled


class BiLSTM_MTL(nn.Module):
    """
    BiLSTM Multi-Task Learning Model
    
    Shared backbone + 2 task-specific heads:
        - AD Head: Binary classification (11 aspects)
        - SC Head: Multi-class classification (11 × 3)
    """
    
    def __init__(
        self,
        vocab_size=30000,
        embedding_dim=300,
        num_aspects=11,
        num_sentiments=3,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        lstm_dropout=0.3,
        spatial_dropout=0.2,
        conv_filters=128,
        conv_kernel_size=3,
        dense_hidden_size=256,
        dense_dropout=0.3,
        padding_idx=0,
        use_phobert_embeddings=False,
        freeze_embeddings=True,
        phobert_model_name=None
    ):
        """
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension (trainable)
            num_aspects: Number of aspects (11)
            num_sentiments: Number of sentiments (3)
            lstm_hidden_size: BiLSTM hidden size
            lstm_num_layers: Number of BiLSTM layers
            lstm_dropout: BiLSTM dropout
            spatial_dropout: Spatial dropout for embeddings
            conv_filters: Conv1D filters
            conv_kernel_size: Conv1D kernel size
            dense_hidden_size: Dense layer size
            dense_dropout: Dense dropout
            padding_idx: Padding token index
            use_phobert_embeddings: If True, load pretrained PhoBERT embeddings
            freeze_embeddings: If True, freeze embedding weights (no gradient)
            phobert_model_name: HuggingFace model name for PhoBERT (e.g. 'vinai/phobert-base-v2')
        """
        super(BiLSTM_MTL, self).__init__()
        
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
        self.embedding_dim = embedding_dim
        
        # =====================================================================
        # SHARED BACKBONE
        # =====================================================================
        
        # 1. Embedding layer (pretrained PhoBERT or random)
        if use_phobert_embeddings and phobert_model_name:
            print(f"[BiLSTM_MTL] Loading pretrained embeddings from {phobert_model_name}...")
            from transformers import AutoModel
            phobert_model = AutoModel.from_pretrained(phobert_model_name)
            phobert_weights = phobert_model.embeddings.word_embeddings.weight.data.clone()
            del phobert_model
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            vocab_size = phobert_weights.shape[0]
            embedding_dim = phobert_weights.shape[1]
            self.embedding_dim = embedding_dim
            
            self.embeddings = nn.Embedding.from_pretrained(
                phobert_weights,
                freeze=freeze_embeddings,
                padding_idx=padding_idx
            )
            freeze_str = "FROZEN" if freeze_embeddings else "TRAINABLE"
            print(f"[BiLSTM_MTL] ✅ PhoBERT embeddings loaded: {vocab_size} × {embedding_dim} ({freeze_str})")
        else:
            self.embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx
            )
            print(f"[BiLSTM_MTL] Using random embeddings: {vocab_size} × {embedding_dim} (TRAINABLE)")
        
        # 2. Spatial Dropout
        self.spatial_dropout = nn.Dropout2d(p=spatial_dropout)
        
        # 3. BiLSTM
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # NEW: MultiHead Self-Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.attn_layer_norm = nn.LayerNorm(lstm_hidden_size * 2)
        
        # 4. Conv1D
        self.conv1d = nn.Conv1d(
            in_channels=lstm_hidden_size * 2,  # Bidirectional
            out_channels=conv_filters,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2
        )
        
        # 5. Global Pooling
        self.global_pooling = GlobalPooling()
        
        # 6. Shared dense layer
        self.shared_dense = nn.Linear(conv_filters * 2, dense_hidden_size)
        self.shared_dropout = nn.Dropout(dense_dropout)
        self.shared_activation = nn.ReLU()
        
        # =====================================================================
        # TASK-SPECIFIC HEADS
        # =====================================================================
        
        # Head 1: Aspect Detection (Binary classification)
        self.ad_dense = nn.Linear(dense_hidden_size, dense_hidden_size // 2)
        self.ad_dropout = nn.Dropout(dense_dropout)
        self.ad_activation = nn.ReLU()
        self.ad_output = nn.Linear(dense_hidden_size // 2, num_aspects)  # Binary logits
        
        # Head 2: Sentiment Classification (Multi-class)
        self.sc_dense = nn.Linear(dense_hidden_size, dense_hidden_size // 2)
        self.sc_dropout = nn.Dropout(dense_dropout)
        self.sc_activation = nn.ReLU()
        self.sc_output = nn.Linear(dense_hidden_size // 2, num_aspects * num_sentiments)
    
    def forward(self, input_ids, attention_mask, return_both=True):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_both: If True, return both AD and SC outputs
        
        Returns:
            If return_both=True:
                ad_logits: [batch_size, num_aspects] (binary logits)
                sc_logits: [batch_size, num_aspects, num_sentiments] (multi-class logits)
            If return_both=False:
                Tuple of (ad_logits, sc_logits)
        """
        batch_size = input_ids.size(0)
        
        # =====================================================================
        # SHARED BACKBONE
        # =====================================================================
        
        # 1. Embedding
        x = self.embeddings(input_ids)  # [batch, seq_len, embedding_dim]
        
        # 2. Spatial Dropout
        x = x.unsqueeze(2)  # [batch, seq_len, 1, embedding_dim]
        x = self.spatial_dropout(x)
        x = x.squeeze(2)  # [batch, seq_len, embedding_dim]
        
        # 3. BiLSTM
        lstm_out, _ = self.bilstm(x)  # [batch, seq_len, lstm_hidden*2]
        
        # NEW: MultiHead Self-Attention
        # Query, Key, Value are all lstm_out for self-attention
        # attn_out shape is same as lstm_out: [batch, seq_len, lstm_hidden*2]
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)
        
        # Add & Norm (Residual connection)
        lstm_out = self.attn_layer_norm(lstm_out + attn_out)
        
        # 4. Conv1D (need to transpose for Conv1D)
        conv_input = lstm_out.transpose(1, 2)  # [batch, lstm_hidden*2, seq_len]
        conv_out = self.conv1d(conv_input)  # [batch, conv_filters, seq_len]
        conv_out = F.relu(conv_out)
        conv_out = conv_out.transpose(1, 2)  # [batch, seq_len, conv_filters]
        
        # 5. Global Pooling
        pooled = self.global_pooling(conv_out)  # [batch, conv_filters*2]
        
        # 6. Shared dense layer
        shared = self.shared_dense(pooled)
        shared = self.shared_activation(shared)
        shared = self.shared_dropout(shared)  # [batch, dense_hidden_size]
        
        # =====================================================================
        # TASK-SPECIFIC HEADS
        # =====================================================================
        
        # Head 1: Aspect Detection (Binary)
        ad_hidden = self.ad_dense(shared)
        ad_hidden = self.ad_activation(ad_hidden)
        ad_hidden = self.ad_dropout(ad_hidden)
        ad_logits = self.ad_output(ad_hidden)  # [batch, num_aspects]
        
        # Head 2: Sentiment Classification (Multi-class)
        sc_hidden = self.sc_dense(shared)
        sc_hidden = self.sc_activation(sc_hidden)
        sc_hidden = self.sc_dropout(sc_hidden)
        sc_flat = self.sc_output(sc_hidden)  # [batch, num_aspects * num_sentiments]
        sc_logits = sc_flat.view(batch_size, self.num_aspects, self.num_sentiments)
        
        if return_both:
            return ad_logits, sc_logits
        else:
            return ad_logits, sc_logits
    
    def predict_ad(self, input_ids, attention_mask, threshold=0.5):
        """
        Predict aspect detection only
        
        Returns:
            predictions: [batch_size, num_aspects] (binary: 0 or 1)
            probabilities: [batch_size, num_aspects]
        """
        with torch.no_grad():
            ad_logits, _ = self.forward(input_ids, attention_mask)
            probs = torch.sigmoid(ad_logits)
            preds = (probs >= threshold).long()
        return preds, probs
    
    def predict_sc(self, input_ids, attention_mask):
        """
        Predict sentiment classification only
        
        Returns:
            predictions: [batch_size, num_aspects] (class indices)
            probabilities: [batch_size, num_aspects, num_sentiments]
        """
        with torch.no_grad():
            _, sc_logits = self.forward(input_ids, attention_mask)
            probs = F.softmax(sc_logits, dim=-1)
            preds = torch.argmax(sc_logits, dim=-1)
        return preds, probs
    
    def predict_both(self, input_ids, attention_mask, ad_threshold=0.5):
        """
        Predict both tasks
        
        Returns:
            ad_predictions: [batch_size, num_aspects] (binary)
            ad_probs: [batch_size, num_aspects]
            sc_predictions: [batch_size, num_aspects] (class indices)
            sc_probs: [batch_size, num_aspects, num_sentiments]
        """
        with torch.no_grad():
            ad_logits, sc_logits = self.forward(input_ids, attention_mask)
            
            # AD predictions
            ad_probs = torch.sigmoid(ad_logits)
            ad_preds = (ad_probs >= ad_threshold).long()
            
            # SC predictions
            sc_probs = F.softmax(sc_logits, dim=-1)
            sc_preds = torch.argmax(sc_logits, dim=-1)
        
        return ad_preds, ad_probs, sc_preds, sc_probs


def test_model():
    """Test BiLSTM MTL model"""
    print("="*80)
    print("Testing BiLSTM Multi-Task Learning Model")
    print("="*80)
    
    # Create model
    print("\n1. Creating model...")
    model = BiLSTM_MTL(
        vocab_size=30000,
        embedding_dim=300,
        num_aspects=11,
        num_sentiments=3,
        lstm_hidden_size=256,
        lstm_num_layers=2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 4
    seq_len = 128
    
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    model.eval()
    with torch.no_grad():
        ad_logits, sc_logits = model(input_ids, attention_mask)
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   AD logits shape: {ad_logits.shape}  (expected: [{batch_size}, 11])")
    print(f"   SC logits shape: {sc_logits.shape}  (expected: [{batch_size}, 11, 3])")
    
    # Test predictions
    print("\n3. Testing predictions...")
    ad_preds, ad_probs, sc_preds, sc_probs = model.predict_both(input_ids, attention_mask)
    
    print(f"   AD predictions shape: {ad_preds.shape}  (binary)")
    print(f"   AD probabilities shape: {ad_probs.shape}")
    print(f"   SC predictions shape: {sc_preds.shape}  (class indices)")
    print(f"   SC probabilities shape: {sc_probs.shape}")
    
    print("\n   Sample AD predictions: {ad_preds[0]}")
    print(f"   Sample SC predictions: {sc_preds[0]}")
    
    print("\nAll tests passed!")
    print("="*80)


if __name__ == '__main__':
    test_model()

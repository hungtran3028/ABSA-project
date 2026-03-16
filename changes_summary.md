# Summary of ABSA Project Improvements (Batch Review)

This document contains a summary of all recent architectural and configuration changes made to the ABSA Project. Use this for reference when deciding to commit or move to Claude.ai.

## 1. Architectural Upgrades

### Attention Mechanism Integration
- **Transformer Models**: Added `use_attention: true` support to `PhoBERT` and `ViSoBERT` (MTL & STL).
- **Functionality**: The `CLS` token output is now passed through a Multi-Head Attention layer (where appropriate) or a Scale-Dot Product Attention layer to enhance context awareness before the classification head.
- **BiLSTM**: Standardized the Attention-CNN fusion logic to better capture morphological features.

### 2-Stage BiLSTM Training (STL)
- **Problem**: BiLSTM often struggled with sentiment detection once aspect detection was mixed.
- **Solution**: Rewrote `BILSTM-STL/train_two_stage_bilstm.py` to:
    1. Train for Aspect Detection (AD) first.
    2. Freeze the encoder and train specifically for Sentiment Classification (SC) using a separate head.
    3. Automate the transition between stages.

## 2. Configuration & Loss Optimization

### Focal Loss Standardization
- All models now use **Focal Loss** for both Aspect and Sentiment tasks by default.
- **Params**: `gamma: 2.0`, `alpha: controlled by pos_weight`. This significantly improves "Minority Sentiment" (e.g., negative reviews) detection.

### Hyperparameter Alignment
- **Learning Rate**: Standardized to `2e-5` for Transformers and `1e-3` for BiLSTM.
- **Batch Size**: Optimized for 16GB VRAM (Tesla T4).
- **Reproducibility**: `seed: 42` applied across all training pipelines.

## 3. Workflow & Environment

### Unified Data Preparation
- **Script**: `run_data_preparation.sh` now automatically populates **all 6** model directories.
- **Balanced Augmentation**: Integrated `augment_multilabel_balanced.py` as a default step to handle aspect class imbalance.

### Environment Setup
- **`requirements.txt`**: Added `wandb`, `emoji`, and `pyvi`.
- **Kaggle**: Removed redundant data copying steps in the main notebook to reduce cell execution time.

## 4. GitHub & Branching Strategy
- Recommended using a `feature/attention-optimization` branch to keep `main` stable until these changes are verified.
- **Commit Pattern**: Conventional commits (`feat:`, `refactor:`, `config:`) for better traceability.

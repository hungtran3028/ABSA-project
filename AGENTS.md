# ABSA Project - Aspect-Based Sentiment Analysis

Vietnamese Sentiment Analysis project focusing on smartphone reviews.

## Architecture
- **Models**: PhoBERT, ViSoBERT, BiLSTM-CNN
- **Approaches**: Multi-Task Learning (MTL) and Sequential Single-Task Learning (STL)
- **Environment**: Kaggle GPU (Tesla T4), Python 3.10, PyTorch 2.1+, CUDA 12.1

## Core Rules
- Enable `use_attention: true` in configs for all models.
- Use **Focal Loss** (`gamma: 2.0`) to handle aspect/sentiment imbalance.
- Data prep must use `run_data_preparation.sh` to populate all 6 model directories.
- Log experiments to Wandb project `ABSA-Vietnamese`.

## Common Commands
| Task | Command |
|------|---------|
| Data Prep | `bash run_data_preparation.sh` |
| Train ViSoBERT MTL | `python VisoBERT-MTL/train_visobert_mtl.py --config VisoBERT-MTL/config_visobert_mtl.yaml` |
| Train BiLSTM STL | `python BILSTM-STL/train_two_stage_bilstm.py --config BILSTM-STL/config_bilstm_stl.yaml` |

## Model Status
1. **ViSoBERT-MTL**: Attention integrated, Focal Loss enabled.
2. **ViSoBERT-STL**: 2-stage training with Attention.
3. **PhoBERT-MTL**: Attention integrated.
4. **PhoBERT-STL**: Sequential training with Attention.
5. **BiLSTM-MTL**: Built-in Attention + CNN.
6. **BiLSTM-STL**: Rewritten 2-stage logic with Attention support.

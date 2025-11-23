# Sparse Autoencoder (SAE) Training

Implementation of the Sparse Autoencoder from **"I Have Covered All the Bases Here: Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders"** (arXiv:2503.18878v2).

## Quick Start

### 1. Test the SAE

```bash
cd autoencoder
python3 test_autoencoder.py
```

This will:
- Test the single SAE on dummy data
- Test the layer-wise SAE
- Test loading your saved activations (if available)

### 2. Train the SAE

```bash
python3 train_autoencoder.py
```

Or with custom settings:
```bash
python3 train_autoencoder.py --num_epochs 20 --device cuda
```

## What Changed from Standard SAE

### Key Formula (Equation 2 from paper)

**Loss:** `L = ||x - x̂||²₂ + λ Σᵢ fᵢ ||W_dec,i||₂`

The critical difference is the **modified L1 penalty** where each latent activation `fᵢ` is weighted by the L2 norm of its decoder weight vector `||W_dec,i||₂`. This prevents trivial solutions.

### Hyperparameters (from paper Section 4.1)

```python
ACTIVATION_DIM = 4096      # LLaMA-8B hidden size (n)
LATENT_DIM = 65536         # 16x overcomplete expansion (m)
SPARSITY_COEFFICIENT = 5.0 # λ (strong sparsity)
LEARNING_RATE = 5e-5       # η
BATCH_SIZE = 4096          # Large batches
GRADIENT_CLIP_NORM = 1.0   # Max gradient norm
```

## Architecture

```
Input (4096)  →  Encoder  →  Latent (65536)  →  Decoder  →  Reconstructed (4096)
                  ↓                                ↓
              f = ReLU(W_enc·x + b_enc)      x̂ = W_dec·f + b_dec
```

## Expected Results (after proper training)

| Metric | Target |
|--------|--------|
| L0 norm | 86 (avg active features) |
| Variance explained | 68.5% |

This means ~86 out of 65,536 features are active per input (0.13% sparsity) while preserving 68.5% of variance.

## Files

- `sparse_autoencoder.py` - SAE implementation matching paper
- `train_autoencoder.py` - Training script with gradient clipping
- `test_autoencoder.py` - Testing and validation
- `config.py` - Hyperparameters from paper

## Training Tips

1. **Start small**: Test with a few activation batches first
2. **Monitor L0**: Should decrease toward ~86 during training
3. **Check variance**: Should increase toward 68.5%
4. **Gradient clipping**: Already enabled (max norm = 1.0)
5. **Save checkpoints**: Best model saved automatically

## What the Paper Used

- **Model**: DeepSeek-R1-LLaMA-8B (layer 19)
- **Data**: 1B tokens (500M LMSYS-CHAT-1M + 500M OPENTHOUGHTS-114K)
- **Training**: Adam optimizer, 5% sparsity warmup, 20% LR decay

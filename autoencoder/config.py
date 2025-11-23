"""
Configuration for Sparse Autoencoder training.

Based on "I Have Covered All the Bases Here: Interpreting Reasoning Features in
Large Language Models via Sparse Autoencoders" (2025), Section 4.1.
"""

# ============================================================================
# Model Architecture (Section 4.1)
# ============================================================================
ACTIVATION_DIM = 4096  # Hidden dimension (n) of DeepSeek-R1-Distill-Llama-8B
LATENT_DIM = 65536  # Sparse overcomplete dimension (m = 16 × n, as in paper)
SPARSITY_COEFFICIENT = 5.0  # λ - sparsity penalty (paper uses linear warmup to λ=5)
TIE_WEIGHTS = False  # Whether to tie encoder and decoder weights

# ============================================================================
# Training Settings (Section 4.1 - following Anthropic April update)
# ============================================================================
# Optimizer: Adam with β1=0.9, β2=0.999
LEARNING_RATE = 5e-5  # η from paper
BATCH_SIZE = 4096  # Batch size from paper
NUM_EPOCHS = 10  # Adjust based on your data
WEIGHT_DECAY = 0.0  # Not used in paper

# Learning rate schedule
USE_LR_DECAY = True  # Linear decay to 0 over last 20% of training
LR_DECAY_START = 0.8  # Start decay at 80% of training

# Sparsity coefficient warmup (Section 4.1)
USE_SPARSITY_WARMUP = True  # Linear warmup from 0 to λ
SPARSITY_WARMUP_STEPS_RATIO = 0.05  # Warmup over first 5% of training steps

# Gradient clipping (Section 4.1)
GRADIENT_CLIP_NORM = 1.0  # Clip gradient norm to 1

# ============================================================================
# Data Settings
# ============================================================================
ACTIVATION_DIR = "teacher/activations"  # Where teacher activations are saved
CHECKPOINT_DIR = "autoencoder/checkpoints"  # Where to save trained models
LOG_DIR = "autoencoder/logs"  # Training logs

# Training data (Section 4.1)
# Paper trains on 1B tokens: 500M from LMSYS-CHAT-1M + 500M from OPENTHOUGHTS-114K
# With context window of 1024 tokens
CONTEXT_WINDOW = 1024  # Tokens per sequence
TOTAL_TOKENS = 1_000_000_000  # 1B tokens total
NUM_SEQUENCES = TOTAL_TOKENS // CONTEXT_WINDOW  # ~976,563 sequences

# ============================================================================
# Logging and Checkpointing
# ============================================================================
LOG_INTERVAL = 100  # Print stats every N batches
SAVE_INTERVAL = 10000  # Save checkpoint every N batches
SAVE_BEST = True  # Save best model based on validation loss

# Evaluation metrics (Section 4.1)
# Paper reports: L0=86, variance explained=68.5%
TARGET_L0 = 86
TARGET_VARIANCE_EXPLAINED = 0.685

# ============================================================================
# Device
# ============================================================================
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

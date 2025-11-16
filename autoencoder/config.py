"""
Configuration for Sparse Autoencoder training.
"""

# Model architecture
ACTIVATION_DIM = 4096  # Hidden dimension of DeepSeek-R1-Distill-Llama-8B
LATENT_DIM = 512  # Compressed latent dimension (8x compression)
SPARSITY_COEFFICIENT = 1e-3  # L1 penalty weight for sparsity
TIE_WEIGHTS = False  # Whether to tie encoder and decoder weights

# Training settings
LEARNING_RATE = 1e-3
BATCH_SIZE = 32  # How many activation batches to load at once
NUM_EPOCHS = 10
WEIGHT_DECAY = 1e-5

# Data settings
ACTIVATION_DIR = "teacher/activations"  # Where teacher activations are saved
CHECKPOINT_DIR = "autoencoder/checkpoints"  # Where to save trained models
LOG_DIR = "autoencoder/logs"  # Training logs

# Logging and checkpointing
LOG_INTERVAL = 10  # Print stats every N batches
SAVE_INTERVAL = 100  # Save checkpoint every N batches
SAVE_BEST = True  # Save best model based on validation loss

# Device
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

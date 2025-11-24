"""
Configuration file for teacher model setup and activation capture.
"""

# Model configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Data configuration
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 2048

# Layers to hook for activation capture
# DeepSeek-R1-Distill-Llama-8B has 32 layers (0-31)
# Hook every 4th layer for memory efficiency
LAYERS_TO_HOOK = [16]  # middle layer

# Output directory for activations
ACTIVATION_DIR = "teacher/activations"

# Device configuration
# Will be determined at runtime based on available resources
DEVICE = "auto"  # Options: "auto", "cuda", "cpu"
DTYPE = "auto"  # Options: "auto", "bfloat16", "float32"

# Offloading configuration
USE_DISK_OFFLOAD = False  # Set to True if memory is limited
OFFLOAD_FOLDER = "teacher/offload"


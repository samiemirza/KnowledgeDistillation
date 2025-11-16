"""
Configuration file for teacher model setup and activation capture.
"""

# Model configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Data configuration
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 512

# Layers to hook for activation capture
# Adjust based on the model architecture (DeepSeek-R1-Distill-Llama-8B has 32 layers)
LAYERS_TO_HOOK = [0, 8, 16, 24, 31]  # First, middle, and last layers

# Output directory for activations
ACTIVATION_DIR = "teacher/activations"

# Device configuration
# Will be determined at runtime based on available resources
DEVICE = "auto"  # Options: "auto", "cuda", "cpu"
DTYPE = "auto"  # Options: "auto", "bfloat16", "float32"

# Offloading configuration
USE_DISK_OFFLOAD = False  # Set to True if memory is limited
OFFLOAD_FOLDER = "teacher/offload"

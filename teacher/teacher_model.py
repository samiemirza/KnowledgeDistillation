"""
Teacher model loader for DeepSeek-R1-Distill-Llama-8B.
Handles model loading with proper device placement and dtype configuration.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config
import os


def get_device_and_dtype():
    """
    Determine the optimal device and dtype based on available resources.

    Returns:
        tuple: (device, dtype)
    """
    if config.DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.DEVICE

    if config.DTYPE == "auto":
        # Use bfloat16 on GPU if available, float32 on CPU
        if device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    elif config.DTYPE == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    return device, dtype


def load_teacher_model():
    """
    Load the DeepSeek-R1-Distill-Llama-8B teacher model from Hugging Face.

    Returns:
        tuple: (model, tokenizer, device, dtype)
    """
    device, dtype = get_device_and_dtype()

    print(f"Loading teacher model: {config.MODEL_NAME}")
    print(f"Device: {device}, Dtype: {dtype}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with appropriate configuration
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }

    # Configure offloading if needed
    if config.USE_DISK_OFFLOAD:
        os.makedirs(config.OFFLOAD_FOLDER, exist_ok=True)
        model_kwargs["device_map"] = "auto"
        model_kwargs["offload_folder"] = config.OFFLOAD_FOLDER
        model_kwargs["offload_state_dict"] = True
        print(f"Using disk offloading to: {config.OFFLOAD_FOLDER}")
    elif device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        **model_kwargs
    )

    # Move to CPU if not using device_map
    if device == "cpu" and not config.USE_DISK_OFFLOAD:
        model = model.to(device)

    model.eval()  # Set to evaluation mode

    print(f"Model loaded successfully")
    print(f"Model has {model.config.num_hidden_layers} layers")

    return model, tokenizer, device, dtype


def get_layer_modules(model):
    """
    Get the layer modules from the model for hooking.

    Args:
        model: The loaded model

    Returns:
        dict: Dictionary mapping layer indices to layer modules
    """
    # For Llama-based models, layers are typically in model.model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        return {i: layer for i, layer in enumerate(layers)}
    else:
        raise AttributeError("Could not find model layers. Check model architecture.")


if __name__ == "__main__":
    # Test loading
    model, tokenizer, device, dtype = load_teacher_model()
    layer_modules = get_layer_modules(model)
    print(f"Available layers: {list(layer_modules.keys())}")
    print(f"Layers to hook: {config.LAYERS_TO_HOOK}")

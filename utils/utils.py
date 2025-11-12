import torch.nn as nn


def get_module(model, layer_name):
    """
    Retrieve a specific module from a model using a dot-separated path.
    
    This function navigates through nested modules to find the target layer.
    For example, 'transformer.h.6.mlp' would retrieve the MLP module from
    the 6th transformer block.
    
    Args:
        model (nn.Module): The model to search through.
        layer_name (str): Dot-separated path to the target module.
                         Examples:
                         - 'transformer.h.6' (GPT-2 6th block)
                         - 'model.layers.12.self_attn' (LLaMA 12th layer attention)
                         - 'encoder.layer.5' (BERT 5th encoder layer)
    
    Returns:
        nn.Module: The requested module.
        
    Raises:
        AttributeError: If the layer path is invalid or module doesn't exist.
        
    Examples:
        >>> from transformers import GPT2Model
        >>> model = GPT2Model.from_pretrained('gpt2')
        >>> layer = get_module(model, 'transformer.h.6')
        >>> print(type(layer))  # <class 'transformers.models.gpt2.modeling_gpt2.GPT2Block'>
    """
    # Split the layer name by dots to get the path
    parts = layer_name.split('.')
    
    # Start from the model
    module = model
    
    # Navigate through the path
    for part in parts:
        # Check if the part is an integer (for indexed access like h.6)
        try:
            # Try to convert to integer for list/sequential access
            idx = int(part)
            module = module[idx]
        except (ValueError, TypeError):
            # Not an integer, use attribute access
            if not hasattr(module, part):
                raise AttributeError(
                    f"Module '{type(module).__name__}' has no attribute '{part}'. "
                    f"Full path: '{layer_name}'"
                )
            module = getattr(module, part)
    
    return module


def count_params(model):
    """
    Count the total number of parameters in a PyTorch model or module.
    
    This function counts all parameters, regardless of whether they require
    gradients or not.
    
    Args:
        model (nn.Module): The model or module to count parameters for.
    
    Returns:
        int: Total number of parameters.
        
    Examples:
        >>> linear = nn.Linear(100, 50)
        >>> count_params(linear)
        5050  # 100*50 + 50 (weights + bias)
        
        >>> from transformers import GPT2LMHeadModel
        >>> model = GPT2LMHeadModel.from_pretrained('gpt2')
        >>> print(f"GPT-2 has {count_params(model):,} parameters")
        GPT-2 has 124,439,808 parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model):
    """
    Count only the trainable parameters in a PyTorch model or module.
    
    This function counts only parameters where requires_grad=True.
    
    Args:
        model (nn.Module): The model or module to count parameters for.
    
    Returns:
        int: Total number of trainable parameters.
        
    Examples:
        >>> model = nn.Linear(100, 50)
        >>> # Freeze the model
        >>> for param in model.parameters():
        ...     param.requires_grad = False
        >>> count_trainable_params(model)
        0
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model):
    """
    Get a summary of model parameters including total and trainable counts.
    
    Args:
        model (nn.Module): The model to summarize.
    
    Returns:
        dict: Dictionary containing parameter statistics.
        
    Examples:
        >>> from transformers import GPT2LMHeadModel
        >>> model = GPT2LMHeadModel.from_pretrained('gpt2')
        >>> summary = get_model_summary(model)
        >>> print(f"Total: {summary['total']:,}")
        >>> print(f"Trainable: {summary['trainable']:,}")
        >>> print(f"Frozen: {summary['frozen']:,}")
    """
    total = count_params(model)
    trainable = count_trainable_params(model)
    frozen = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'trainable_percent': (trainable / total * 100) if total > 0 else 0
    }


def print_model_summary(model, model_name="Model"):
    """
    Print a formatted summary of model parameters.
    
    Args:
        model (nn.Module): The model to summarize.
        model_name (str): Name of the model for display.
        
    Examples:
        >>> from transformers import GPT2LMHeadModel
        >>> model = GPT2LMHeadModel.from_pretrained('gpt2')
        >>> print_model_summary(model, "GPT-2")
        GPT-2 Parameter Summary:
        ==================================================
        Total parameters:      124,439,808
        Trainable parameters:  124,439,808
        Frozen parameters:     0
        Trainable percentage:  100.00%
        ==================================================
    """
    summary = get_model_summary(model)
    
    print(f"\n{model_name} Parameter Summary:")
    print("=" * 50)
    print(f"Total parameters:      {summary['total']:,}")
    print(f"Trainable parameters:  {summary['trainable']:,}")
    print(f"Frozen parameters:     {summary['frozen']:,}")
    print(f"Trainable percentage:  {summary['trainable_percent']:.2f}%")
    print("=" * 50 + "\n")


def find_layers_by_type(model, layer_type, prefix=''):
    """
    Find all layers of a specific type in a model and return their names.
    
    Useful for identifying which layers to use for feature extraction.
    
    Args:
        model (nn.Module): The model to search.
        layer_type (type or tuple of types): The layer type(s) to find.
                                            e.g., nn.Linear, nn.LayerNorm, etc.
        prefix (str): Internal use for tracking nested path.
    
    Returns:
        list of tuples: List of (layer_name, layer_module) pairs.
        
    Examples:
        >>> from transformers import GPT2Model
        >>> import torch.nn as nn
        >>> model = GPT2Model.from_pretrained('gpt2')
        >>> # Find all LayerNorm layers
        >>> layernorms = find_layers_by_type(model, nn.LayerNorm)
        >>> print(f"Found {len(layernorms)} LayerNorm layers")
        >>> print(layernorms[0][0])  # Print first LayerNorm's name
    """
    layers = []
    
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        # Check if this module matches the type
        if isinstance(module, layer_type):
            layers.append((full_name, module))
        
        # Recursively search in child modules
        layers.extend(find_layers_by_type(module, layer_type, full_name))
    
    return layers


def list_all_layers(model, max_depth=None, prefix='', current_depth=0):
    """
    List all layers in a model with their names and types.
    
    Useful for exploring model architecture and finding layer names for hooks.
    
    Args:
        model (nn.Module): The model to explore.
        max_depth (int, optional): Maximum depth to traverse. None for unlimited.
        prefix (str): Internal use for tracking nested path.
        current_depth (int): Internal use for tracking depth.
    
    Returns:
        list of tuples: List of (layer_name, layer_type) pairs.
        
    Examples:
        >>> from transformers import GPT2Model
        >>> model = GPT2Model.from_pretrained('gpt2')
        >>> layers = list_all_layers(model, max_depth=3)
        >>> for name, layer_type in layers[:10]:
        ...     print(f"{name}: {layer_type}")
    """
    layers = []
    
    # Check depth limit
    if max_depth is not None and current_depth >= max_depth:
        return layers
    
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        layer_type = type(module).__name__
        
        layers.append((full_name, layer_type))
        
        # Recursively explore child modules
        layers.extend(
            list_all_layers(
                module, 
                max_depth, 
                full_name, 
                current_depth + 1
            )
        )
    
    return layers


def print_layer_structure(model, max_depth=3, model_name="Model"):
    """
    Print a formatted view of the model's layer structure.
    
    Args:
        model (nn.Module): The model to display.
        max_depth (int): Maximum depth to display.
        model_name (str): Name of the model for display.
        
    Examples:
        >>> from transformers import GPT2Model
        >>> model = GPT2Model.from_pretrained('gpt2')
        >>> print_layer_structure(model, max_depth=2, model_name="GPT-2")
    """
    print(f"\n{model_name} Layer Structure:")
    print("=" * 70)
    
    layers = list_all_layers(model, max_depth=max_depth)
    
    for name, layer_type in layers:
        # Calculate indentation based on depth
        depth = name.count('.')
        indent = "  " * depth
        
        # Get just the last part of the name
        display_name = name.split('.')[-1]
        
        print(f"{indent}├─ {display_name}: {layer_type}")
    
    print("=" * 70 + "\n")

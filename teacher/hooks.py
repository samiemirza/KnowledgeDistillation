"""
Activation hooks for capturing layer outputs during forward pass.
"""

import torch
import os
from pathlib import Path
import config


class ActivationStore:
    """
    Store activations captured from forward hooks.
    Saves activations to disk after each batch.
    """

    def __init__(self, save_dir=None):
        """
        Initialize the activation store.

        Args:
            save_dir (str): Directory to save activations. Defaults to config.ACTIVATION_DIR
        """
        self.save_dir = save_dir or config.ACTIVATION_DIR
        os.makedirs(self.save_dir, exist_ok=True)

        # Storage for current batch activations
        self.activations = {}
        self.batch_idx = 0

        # Hook handles for cleanup
        self.hook_handles = []

    def get_hook_fn(self, layer_name):
        """
        Create a hook function for a specific layer.

        Args:
            layer_name (str): Name/identifier for the layer

        Returns:
            function: Hook function to be registered
        """
        def hook_fn(module, input, output):
            """
            Forward hook function that captures the output activations.

            Args:
                module: The layer module
                input: Input to the layer
                output: Output from the layer (activations to capture)
            """
            # Handle different output types (tuple, tensor, etc.)
            if isinstance(output, tuple):
                activation = output[0]  # Usually the first element is the activation
            else:
                activation = output

            # Detach and move to CPU to save memory
            # Store as float32 to save space (bfloat16 not always supported for saving)
            self.activations[layer_name] = activation.detach().cpu().float()

        return hook_fn

    def save_batch(self):
        """
        Save the current batch of activations to disk.
        """
        if not self.activations:
            print("Warning: No activations to save")
            return

        # Create filename for this batch
        save_path = os.path.join(self.save_dir, f"batch_{self.batch_idx:06d}.pt")

        # Save all activations for this batch
        torch.save(self.activations, save_path)
        print(f"Saved activations for batch {self.batch_idx} to {save_path}")

        # Clear activations for next batch
        self.activations = {}
        self.batch_idx += 1

    def clear(self):
        """
        Clear stored activations without saving.
        """
        self.activations = {}

    def remove_hooks(self):
        """
        Remove all registered hooks.
        """
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        print(f"Removed {len(self.hook_handles)} hooks")


def attach_hooks(model, layer_modules, layers_to_hook, activation_store):
    """
    Attach forward hooks to specified layers of the model.

    Args:
        model: The model to attach hooks to
        layer_modules (dict): Dictionary mapping layer indices to modules
        layers_to_hook (list): List of layer indices to attach hooks to
        activation_store (ActivationStore): Store for captured activations

    Returns:
        ActivationStore: The activation store with hooks attached
    """
    print(f"Attaching hooks to layers: {layers_to_hook}")

    for layer_idx in layers_to_hook:
        if layer_idx not in layer_modules:
            print(f"Warning: Layer {layer_idx} not found in model. Skipping.")
            continue

        layer_module = layer_modules[layer_idx]
        layer_name = f"layer_{layer_idx}"

        # Create and register the hook
        hook_fn = activation_store.get_hook_fn(layer_name)
        handle = layer_module.register_forward_hook(hook_fn)

        # Store handle for later removal
        activation_store.hook_handles.append(handle)

        print(f"  Attached hook to {layer_name}")

    print(f"Successfully attached {len(activation_store.hook_handles)} hooks")

    return activation_store


if __name__ == "__main__":
    # Test activation store
    store = ActivationStore()
    print(f"Activation store initialized. Save directory: {store.save_dir}")

    # Test storing dummy activations
    store.activations["layer_0"] = torch.randn(4, 512, 4096)
    store.activations["layer_8"] = torch.randn(4, 512, 4096)
    store.save_batch()

    print("Test completed successfully")

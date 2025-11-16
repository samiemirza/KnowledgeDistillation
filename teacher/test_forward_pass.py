"""
Test script to verify the teacher model works with a forward pass.
Captures activations and displays them to verify everything is working.
"""

import torch
from teacher_model import load_teacher_model, get_layer_modules
from hooks import ActivationStore, attach_hooks
import config


def test_forward_pass():
    """
    Run a simple forward pass to test the setup.
    """
    print("="*80)
    print("TESTING TEACHER MODEL FORWARD PASS")
    print("="*80)

    # Load model and tokenizer
    print("\n[1/4] Loading model...")
    model, tokenizer, device, dtype = load_teacher_model()

    # Get layer modules
    print("\n[2/4] Getting layer modules...")
    layer_modules = get_layer_modules(model)
    print(f"Found {len(layer_modules)} layers")
    print(f"Will hook layers: {config.LAYERS_TO_HOOK}")

    # Setup activation capture
    print("\n[3/4] Setting up activation hooks...")
    activation_store = ActivationStore(save_dir="teacher/test_activations")
    attach_hooks(model, layer_modules, config.LAYERS_TO_HOOK, activation_store)

    # Create test input
    print("\n[4/4] Running forward pass...")
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "PyTorch is a popular deep learning framework."
    ]

    print(f"\nTest inputs ({len(test_texts)} samples):")
    for i, text in enumerate(test_texts):
        print(f"  {i+1}. {text}")

    # Tokenize
    inputs = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
        return_tensors='pt'
    )

    # Move to device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    print(f"\nTokenized input shape: {input_ids.shape}")
    print(f"  Batch size: {input_ids.shape[0]}")
    print(f"  Sequence length: {input_ids.shape[1]}")

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

    print("✓ Forward pass completed successfully!")

    # Check captured activations
    print("\n" + "="*80)
    print("CAPTURED ACTIVATIONS")
    print("="*80)

    for layer_name, activation in activation_store.activations.items():
        print(f"\n{layer_name}:")
        print(f"  Shape: {activation.shape}")
        print(f"  Dtype: {activation.dtype}")
        print(f"  Device: {activation.device}")
        print(f"  Min value: {activation.min().item():.4f}")
        print(f"  Max value: {activation.max().item():.4f}")
        print(f"  Mean value: {activation.mean().item():.4f}")

    # Optionally save the test activations
    print("\n" + "="*80)
    save_choice = input("Save test activations to disk? (y/n): ").strip().lower()
    if save_choice == 'y':
        activation_store.save_batch()
        print(f"✓ Saved to teacher/test_activations/batch_000000.pt")
    else:
        print("Activations not saved (just tested in memory)")

    # Cleanup
    activation_store.remove_hooks()

    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n✓ Model loads correctly")
    print("✓ Forward pass works")
    print("✓ Activations are captured")
    print("✓ Ready to process your dataset with dump_activations.py")

    return activation_store.activations


if __name__ == "__main__":
    activations = test_forward_pass()

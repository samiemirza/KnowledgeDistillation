"""
Main script for dumping teacher model activations.
Loads the model, processes data, and saves layer activations batch by batch.
"""

import torch
import argparse
from tqdm import tqdm
import os

import config
from teacher_model import load_teacher_model, get_layer_modules
from hooks import ActivationStore, attach_hooks
from dataset import create_dataloader


def dump_activations(data_file, output_dir=None, num_batches=None):
    """
    Dump activations from the teacher model for a dataset.

    Args:
        data_file (str): Path to the input text file
        output_dir (str): Directory to save activations. Defaults to config.ACTIVATION_DIR
        num_batches (int): Maximum number of batches to process. None means process all.
    """
    # Set output directory
    if output_dir:
        config.ACTIVATION_DIR = output_dir
        os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("TEACHER MODEL ACTIVATION DUMPING")
    print("="*80)

    # Load model and tokenizer
    print("\n[1/5] Loading teacher model...")
    model, tokenizer, device, dtype = load_teacher_model()

    # Get layer modules
    print("\n[2/5] Getting layer modules...")
    layer_modules = get_layer_modules(model)
    print(f"Model has {len(layer_modules)} layers")
    print(f"Will hook into layers: {config.LAYERS_TO_HOOK}")

    # Create activation store
    print("\n[3/5] Setting up activation capture...")
    activation_store = ActivationStore(save_dir=config.ACTIVATION_DIR)

    # Attach hooks to the model
    attach_hooks(model, layer_modules, config.LAYERS_TO_HOOK, activation_store)

    # Create dataloader
    print("\n[4/5] Loading dataset...")
    dataloader = create_dataloader(data_file, tokenizer, batch_size=config.BATCH_SIZE)

    # Process batches
    print("\n[5/5] Processing batches and saving activations...")
    print(f"Output directory: {config.ACTIVATION_DIR}")
    print("-"*80)

    total_batches = num_batches if num_batches else len(dataloader)
    processed_batches = 0

    with torch.no_grad():  # No gradients needed for inference
        for batch_idx, batch in enumerate(tqdm(dataloader, total=total_batches, desc="Processing")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass (hooks will capture activations)
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,  # We're using hooks instead
                    return_dict=True
                )
            except Exception as e:
                print(f"\nError processing batch {batch_idx}: {e}")
                continue

            # Save activations for this batch
            activation_store.save_batch()
            processed_batches += 1

            # Check if we've reached the limit
            if num_batches and processed_batches >= num_batches:
                print(f"\nReached batch limit ({num_batches}). Stopping.")
                break

            # Optional: Clear CUDA cache periodically to prevent OOM
            if device == "cuda" and batch_idx % 10 == 0:
                torch.cuda.empty_cache()

    # Cleanup
    print("\n" + "="*80)
    print("DUMPING COMPLETE")
    print("="*80)
    print(f"Processed {processed_batches} batches")
    print(f"Activations saved to: {config.ACTIVATION_DIR}")

    # Remove hooks
    activation_store.remove_hooks()

    return processed_batches


def main():
    """
    Main entry point with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Dump teacher model activations for knowledge distillation"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the input text file (one sample per line)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Directory to save activations (default: {config.ACTIVATION_DIR})"
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (default: all)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=f"Batch size (default: {config.BATCH_SIZE})"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help=f"Maximum sequence length (default: {config.MAX_SEQ_LENGTH})"
    )

    args = parser.parse_args()

    # Override config if specified
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.max_seq_length:
        config.MAX_SEQ_LENGTH = args.max_seq_length

    # Run activation dumping
    dump_activations(
        data_file=args.data_file,
        output_dir=args.output_dir,
        num_batches=args.num_batches
    )


if __name__ == "__main__":
    main()

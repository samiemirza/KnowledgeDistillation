"""
Main script for dumping teacher model activations.
Loads specific HF datasets and saves layer activations.
"""

import torch
import argparse
from tqdm import tqdm
import os
import sys

# Add parent directory to path to allow imports if running from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from teacher_model import load_teacher_model, get_layer_modules
from hooks import ActivationStore, attach_hooks
from dataset import create_dataloader

def dump_activations(dataset_name, output_dir=None, num_batches=None, max_samples=None):
    """
    Dump activations for a specific dataset.
    """
    # define default output dir based on dataset if not provided
    if output_dir is None:
        output_dir = f"teacher/activations_{dataset_name}"
        # If we are only hooking one layer, maybe append that to path for clarity
        if len(config.LAYERS_TO_HOOK) == 1:
            output_dir = os.path.join(output_dir, f"layer_{config.LAYERS_TO_HOOK[0]}")
    
    config.ACTIVATION_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print(f"TEACHER MODEL ACTIVATION DUMPING: {dataset_name.upper()}")
    print(f"Target Layer(s): {config.LAYERS_TO_HOOK}")
    print("="*80)

    # 1. Load model
    print("\n[1/5] Loading teacher model...")
    model, tokenizer, device, dtype = load_teacher_model()

    # 2. Get layers
    print("\n[2/5] Getting layer modules...")
    layer_modules = get_layer_modules(model)

    # 3. Setup hooks
    print("\n[3/5] Setting up activation capture...")
    activation_store = ActivationStore(save_dir=config.ACTIVATION_DIR)
    attach_hooks(model, layer_modules, config.LAYERS_TO_HOOK, activation_store)

    # 4. Load Dataset
    print(f"\n[4/5] Loading dataset '{dataset_name}'...")
    dataloader = create_dataloader(
        dataset_name, 
        tokenizer, 
        batch_size=config.BATCH_SIZE,
        max_samples=max_samples
    )

    # 5. Process
    print("\n[5/5] Processing batches...")
    print(f"Saving to: {config.ACTIVATION_DIR}")
    
    total_batches = len(dataloader)
    if num_batches: total_batches = min(total_batches, num_batches)
    
    processed_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=total_batches)):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            try:
                # Forward pass triggers hooks
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True
                )
                
                # Save
                activation_store.save_batch()
                processed_batches += 1

                # Limits
                if num_batches and processed_batches >= num_batches:
                    break
                    
            except Exception as e:
                print(f"Error on batch {batch_idx}: {e}")
                continue

    print("\n" + "="*80)
    print("DUMPING COMPLETE")
    print(f"Saved {processed_batches} batches to {config.ACTIVATION_DIR}")
    
    activation_store.remove_hooks()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=['openthoughts', 'lmsys'],
        help="Which dataset to process"
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Limit total samples loaded")
    parser.add_argument("--num_batches", type=int, default=None, help="Stop after N batches")
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    if args.batch_size: config.BATCH_SIZE = args.batch_size

    dump_activations(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        num_batches=args.num_batches
    )

if __name__ == "__main__":
    main() 
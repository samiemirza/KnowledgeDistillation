"""
Training script for Sparse Autoencoder on saved teacher activations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from tqdm import tqdm
import json

import config
from sparse_autoencoder import LayerWiseSparseAutoencoder


class ActivationDataset(Dataset):
    """
    Dataset for loading saved teacher activations from disk.
    """

    def __init__(self, activation_dir):
        """
        Args:
            activation_dir (str): Directory containing saved activation .pt files
        """
        self.activation_dir = activation_dir

        # Find all batch files
        self.batch_files = sorted(Path(activation_dir).glob("batch_*.pt"))

        if len(self.batch_files) == 0:
            raise ValueError(f"No activation files found in {activation_dir}")

        print(f"Found {len(self.batch_files)} activation batch files")

        # Load first batch to get layer information
        first_batch = torch.load(self.batch_files[0])
        self.layer_names = list(first_batch.keys())
        self.layer_dims = {
            name: first_batch[name].shape[-1] for name in self.layer_names
        }

        print(f"Layers found: {self.layer_names}")
        print(f"Layer dimensions: {self.layer_dims}")

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        """
        Load a batch of activations.

        Returns:
            dict: Dictionary mapping layer names to activation tensors
        """
        batch_path = self.batch_files[idx]
        activations = torch.load(batch_path)

        return activations


def train_autoencoder(activation_dir=None, num_epochs=None, device=None):
    """
    Train the sparse autoencoder on saved teacher activations.

    Args:
        activation_dir (str): Directory with saved activations
        num_epochs (int): Number of training epochs
        device (str): Device to train on
    """
    # Use config defaults if not specified
    activation_dir = activation_dir or config.ACTIVATION_DIR
    num_epochs = num_epochs or config.NUM_EPOCHS
    device = device or config.DEVICE

    print("="*80)
    print("TRAINING SPARSE AUTOENCODER")
    print("="*80)

    # Create output directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Load dataset
    print("\n[1/5] Loading activation dataset...")
    dataset = ActivationDataset(activation_dir)

    # Create dataloader (batch_size=1 since each file is already a batch)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Create model
    print("\n[2/5] Creating sparse autoencoder model...")
    model = LayerWiseSparseAutoencoder(
        layer_dims=dataset.layer_dims,
        latent_dim=config.LATENT_DIM,
        sparsity_coefficient=config.SPARSITY_COEFFICIENT,
        tie_weights=config.TIE_WEIGHTS
    ).to(device)

    print(f"Model created with {config.LATENT_DIM}-dim latent space")
    print(f"Sparsity coefficient: {config.SPARSITY_COEFFICIENT}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer (Adam with β1=0.9, β2=0.999 from paper)
    print("\n[3/5] Setting up optimizer...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=config.WEIGHT_DECAY
    )

    # Training loop
    print("\n[4/5] Starting training...")
    print("-"*80)

    best_loss = float('inf')
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_sparsity_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch_activations in enumerate(pbar):
            # Move activations to device
            batch_activations = {
                layer_name: activation.squeeze(0).to(device)  # Remove dataloader batch dim
                for layer_name, activation in batch_activations.items()
            }

            # Forward pass
            results = model(batch_activations)

            # Compute loss
            total_loss, loss_dict = model.compute_total_loss(batch_activations, results)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping (from paper Section 4.1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_NORM)

            optimizer.step()

            # Track losses
            epoch_loss += total_loss.item()

            # Average reconstruction and sparsity losses across layers
            avg_recon = sum(l['reconstruction'] for l in loss_dict.values()) / len(loss_dict)
            avg_sparsity = sum(l['sparsity'] for l in loss_dict.values()) / len(loss_dict)
            epoch_recon_loss += avg_recon
            epoch_sparsity_loss += avg_sparsity

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'recon': f"{avg_recon:.4f}",
                'sparse': f"{avg_sparsity:.4f}"
            })

            global_step += 1

            # Log detailed stats
            if global_step % config.LOG_INTERVAL == 0:
                log_stats = {
                    'epoch': epoch + 1,
                    'step': global_step,
                    'total_loss': total_loss.item(),
                    'avg_recon_loss': avg_recon,
                    'avg_sparsity_loss': avg_sparsity,
                    'per_layer_losses': loss_dict
                }

                # Save to log file
                log_file = os.path.join(config.LOG_DIR, 'training_log.jsonl')
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')

            # Save checkpoint
            if global_step % config.SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join(
                    config.CHECKPOINT_DIR,
                    f"checkpoint_step_{global_step}.pt"
                )
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss.item(),
                }, checkpoint_path)
                print(f"\n✓ Checkpoint saved to {checkpoint_path}")

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_epoch_recon = epoch_recon_loss / len(dataloader)
        avg_epoch_sparsity = epoch_sparsity_loss / len(dataloader)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Average Reconstruction Loss: {avg_epoch_recon:.4f}")
        print(f"  Average Sparsity Loss: {avg_epoch_sparsity:.4f}")

        # Save best model
        if config.SAVE_BEST and avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': {
                    'layer_dims': dataset.layer_dims,
                    'latent_dim': config.LATENT_DIM,
                    'sparsity_coefficient': config.SPARSITY_COEFFICIENT,
                    'tie_weights': config.TIE_WEIGHTS
                }
            }, best_model_path)
            print(f"  ✓ New best model saved! Loss: {best_loss:.4f}")

    # Save final model
    print("\n[5/5] Saving final model...")
    final_model_path = os.path.join(config.CHECKPOINT_DIR, "final_model.pt")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss,
        'config': {
            'layer_dims': dataset.layer_dims,
            'latent_dim': config.LATENT_DIM,
            'sparsity_coefficient': config.SPARSITY_COEFFICIENT,
            'tie_weights': config.TIE_WEIGHTS
        }
    }, final_model_path)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Best loss: {best_loss:.4f}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder")
    parser.add_argument(
        "--activation_dir",
        type=str,
        default=config.ACTIVATION_DIR,
        help="Directory containing saved activations"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=config.NUM_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        help="Device to train on (cuda/cpu)"
    )

    args = parser.parse_args()

    train_autoencoder(
        activation_dir=args.activation_dir,
        num_epochs=args.num_epochs,
        device=args.device
    )

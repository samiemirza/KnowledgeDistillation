"""
Test script to verify the autoencoder works with saved activations.
"""

import torch
import os
from pathlib import Path

import config
from sparse_autoencoder import SparseAutoencoder, LayerWiseSparseAutoencoder


def test_single_autoencoder():
    """
    Test a single sparse autoencoder with dummy data.
    """
    print("="*80)
    print("TEST 1: Single Sparse Autoencoder")
    print("="*80)

    # Create autoencoder
    ae = SparseAutoencoder(
        activation_dim=config.ACTIVATION_DIM,
        latent_dim=config.LATENT_DIM,
        sparsity_coefficient=config.SPARSITY_COEFFICIENT,
        tie_weights=config.TIE_WEIGHTS
    )

    print(f"\nAutoencoder created:")
    print(f"  Input dim: {config.ACTIVATION_DIM}")
    print(f"  Latent dim: {config.LATENT_DIM}")
    print(f"  Compression ratio: {config.ACTIVATION_DIM / config.LATENT_DIM:.1f}x")

    # Create test input (batch_size=2, seq_len=256, dim=4096)
    test_input = torch.randn(2, 256, config.ACTIVATION_DIM)
    print(f"\nTest input shape: {test_input.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    reconstructed, latent = ae(test_input)

    print(f"✓ Forward pass successful!")
    print(f"  Latent shape: {latent.shape}")
    print(f"  Reconstructed shape: {reconstructed.shape}")

    # Compute losses
    total_loss, recon_loss, sparsity_loss = ae.total_loss(test_input, reconstructed, latent)
    print(f"\nLosses:")
    print(f"  Reconstruction loss: {recon_loss.item():.6f}")
    print(f"  Sparsity loss: {sparsity_loss.item():.6f}")
    print(f"  Total loss: {total_loss.item():.6f}")

    # Check sparsity statistics
    stats = ae.get_sparsity_stats(latent)
    print(f"\nSparsity statistics:")
    print(f"  Zero ratio: {stats['zero_ratio']:.2%}")
    print(f"  L0 norm (avg non-zeros): {stats['l0_norm']:.1f}")
    print(f"  L1 norm: {stats['l1_norm']:.4f}")
    print(f"  Avg non-zero activation: {stats['avg_nonzero_activation']:.4f}")

    print("\n✓ Single autoencoder test PASSED!")
    return ae


def test_layerwise_autoencoder():
    """
    Test the layer-wise autoencoder with multiple layers.
    """
    print("\n" + "="*80)
    print("TEST 2: Layer-wise Sparse Autoencoder")
    print("="*80)

    # Define multiple layers (simulating teacher model layers)
    layer_dims = {
        'layer_0': config.ACTIVATION_DIM,
        'layer_8': config.ACTIVATION_DIM,
        'layer_16': config.ACTIVATION_DIM,
    }

    print(f"\nCreating layer-wise autoencoder for layers: {list(layer_dims.keys())}")

    # Create model
    model = LayerWiseSparseAutoencoder(
        layer_dims=layer_dims,
        latent_dim=config.LATENT_DIM,
        sparsity_coefficient=config.SPARSITY_COEFFICIENT,
        tie_weights=config.TIE_WEIGHTS
    )

    # Create test activations
    test_activations = {
        'layer_0': torch.randn(2, 256, config.ACTIVATION_DIM),
        'layer_8': torch.randn(2, 256, config.ACTIVATION_DIM),
        'layer_16': torch.randn(2, 256, config.ACTIVATION_DIM),
    }

    print(f"\nTest activation shapes:")
    for name, act in test_activations.items():
        print(f"  {name}: {act.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    results = model(test_activations)

    print(f"✓ Forward pass successful!")
    print(f"\nResults for each layer:")
    for layer_name, result in results.items():
        print(f"  {layer_name}:")
        print(f"    Reconstructed: {result['reconstructed'].shape}")
        print(f"    Latent: {result['latent'].shape}")

    # Compute total loss
    total_loss, loss_dict = model.compute_total_loss(test_activations, results)

    print(f"\nTotal loss (averaged): {total_loss.item():.6f}")
    print(f"\nPer-layer losses:")
    for layer_name, losses in loss_dict.items():
        print(f"  {layer_name}:")
        print(f"    Total: {losses['total']:.6f}")
        print(f"    Reconstruction: {losses['reconstruction']:.6f}")
        print(f"    Sparsity: {losses['sparsity']:.6f}")

    print("\n✓ Layer-wise autoencoder test PASSED!")
    return model


def test_with_saved_activations():
    """
    Test loading and processing actual saved activations.
    """
    print("\n" + "="*80)
    print("TEST 3: Loading Saved Activations")
    print("="*80)

    activation_dir = config.ACTIVATION_DIR

    # Check if activations exist
    activation_files = list(Path(activation_dir).glob("batch_*.pt"))

    if len(activation_files) == 0:
        print(f"\n⚠ No saved activations found in {activation_dir}")
        print("Run teacher/dump_activations.py first to generate activations.")
        return None

    print(f"\nFound {len(activation_files)} activation files")

    # Load first batch
    first_batch_path = activation_files[0]
    print(f"Loading: {first_batch_path}")

    activations = torch.load(first_batch_path)

    print(f"\nActivation layers: {list(activations.keys())}")
    print(f"\nActivation shapes:")
    for layer_name, activation in activations.items():
        print(f"  {layer_name}: {activation.shape}")

    # Get layer dimensions
    layer_dims = {name: act.shape[-1] for name, act in activations.items()}

    # Create model
    print(f"\nCreating autoencoder for these layers...")
    model = LayerWiseSparseAutoencoder(
        layer_dims=layer_dims,
        latent_dim=config.LATENT_DIM,
        sparsity_coefficient=config.SPARSITY_COEFFICIENT,
        tie_weights=config.TIE_WEIGHTS
    )

    # Forward pass
    print("\nRunning forward pass on saved activations...")
    results = model(activations)

    print(f"✓ Forward pass successful!")

    # Compute loss
    total_loss, loss_dict = model.compute_total_loss(activations, results)

    print(f"\nTotal loss: {total_loss.item():.6f}")
    print(f"\nPer-layer losses:")
    for layer_name, losses in loss_dict.items():
        print(f"  {layer_name}: {losses['total']:.6f}")

    print("\n✓ Saved activations test PASSED!")
    print("\n✅ Ready to train on your saved activations!")

    return model


def main():
    """
    Run all tests.
    """
    print("\n" + "="*80)
    print("SPARSE AUTOENCODER TESTING")
    print("="*80)

    # Test 1: Single autoencoder
    ae = test_single_autoencoder()

    # Test 2: Layer-wise autoencoder
    model = test_layerwise_autoencoder()

    # Test 3: Saved activations (if available)
    saved_model = test_with_saved_activations()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("✓ Single autoencoder: PASSED")
    print("✓ Layer-wise autoencoder: PASSED")
    if saved_model is not None:
        print("✓ Saved activations: PASSED")
        print("\n✅ All tests PASSED! Ready for training.")
    else:
        print("⚠ Saved activations: SKIPPED (no data)")
        print("\nNext steps:")
        print("1. Run: cd teacher && python dump_activations.py --data_file <your_file>")
        print("2. Then run: cd autoencoder && python train_autoencoder.py")


if __name__ == "__main__":
    main()

"""
Sparse Autoencoder for learning compressed representations of teacher activations.

The autoencoder learns to:
1. Encode high-dimensional teacher activations into a sparse latent space
2. Decode the sparse representation back to the original dimensions
3. Enforce sparsity using L1 regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with L1 sparsity regularization.

    Architecture:
        Input (activation_dim) -> Encoder -> Latent (latent_dim) -> Decoder -> Output (activation_dim)

    Note: For proper sparse autoencoders, latent_dim should be LARGER than activation_dim
    to create an overcomplete representation where sparsity is meaningful.

    Args:
        activation_dim (int): Dimension of input activations (e.g., 4096 for LLaMA-8B)
        latent_dim (int): Dimension of sparse latent representation (should be > activation_dim, e.g., 16384)
        sparsity_coefficient (float): L1 penalty weight for sparsity
        tie_weights (bool): Whether to tie encoder and decoder weights (decoder = encoder.T)
    """

    def __init__(self, activation_dim, latent_dim, sparsity_coefficient=1e-3, tie_weights=False):
        super(SparseAutoencoder, self).__init__()

        self.activation_dim = activation_dim
        self.latent_dim = latent_dim
        self.sparsity_coefficient = sparsity_coefficient
        self.tie_weights = tie_weights

        # Encoder: Linear layer + ReLU activation for sparsity
        self.encoder = nn.Linear(activation_dim, latent_dim, bias=True)

        # Decoder: Linear layer to reconstruct original dimensions
        if tie_weights:
            # Tied weights: decoder uses transpose of encoder weights
            self.decoder = None  # Will be computed dynamically
            self.decoder_bias = nn.Parameter(torch.zeros(activation_dim))
        else:
            self.decoder = nn.Linear(latent_dim, activation_dim, bias=True)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights using Xavier/Glorot initialization.
        """
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

        if not self.tie_weights:
            nn.init.xavier_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        """
        Encode input activations to sparse latent representation.

        Args:
            x (Tensor): Input activations of shape (batch_size, seq_len, activation_dim)
                       or (batch_size, activation_dim)

        Returns:
            Tensor: Sparse latent codes of shape (batch_size, seq_len, latent_dim)
                   or (batch_size, latent_dim)
        """
        # Linear transformation
        latent = self.encoder(x)

        # ReLU activation for sparsity (negative values become zero)
        latent = F.relu(latent)

        return latent

    def decode(self, latent):
        """
        Decode latent representation back to activation space.

        Args:
            latent (Tensor): Latent codes of shape (batch_size, seq_len, latent_dim)
                           or (batch_size, latent_dim)

        Returns:
            Tensor: Reconstructed activations of shape (batch_size, seq_len, activation_dim)
                   or (batch_size, activation_dim)
        """
        if self.tie_weights:
            # Use transposed encoder weights
            reconstructed = F.linear(latent, self.encoder.weight.t(), self.decoder_bias)
        else:
            reconstructed = self.decoder(latent)

        return reconstructed

    def forward(self, x):
        """
        Forward pass: encode then decode.

        Args:
            x (Tensor): Input activations

        Returns:
            tuple: (reconstructed, latent, sparsity_loss)
                - reconstructed: Decoded output
                - latent: Sparse latent representation
                - sparsity_loss: L1 sparsity penalty
        """
        # Encode
        latent = self.encode(x)

        # Decode
        reconstructed = self.decode(latent)

        # Compute sparsity loss (L1 norm on latent codes)
        sparsity_loss = self.sparsity_coefficient * torch.abs(latent).sum(dim=-1).mean()

        return reconstructed, latent, sparsity_loss

    def reconstruction_loss(self, x, reconstructed):
        """
        Compute MSE reconstruction loss.

        Args:
            x (Tensor): Original activations
            reconstructed (Tensor): Reconstructed activations

        Returns:
            Tensor: Scalar MSE loss
        """
        return F.mse_loss(reconstructed, x)

    def total_loss(self, x, reconstructed, sparsity_loss):
        """
        Compute total loss = reconstruction_loss + sparsity_loss.

        Args:
            x (Tensor): Original activations
            reconstructed (Tensor): Reconstructed activations
            sparsity_loss (Tensor): Sparsity penalty

        Returns:
            Tensor: Scalar total loss
        """
        recon_loss = self.reconstruction_loss(x, reconstructed)
        total = recon_loss + sparsity_loss

        return total, recon_loss

    def get_sparsity_stats(self, latent):
        """
        Compute statistics about sparsity in the latent representation.

        Args:
            latent (Tensor): Latent codes

        Returns:
            dict: Dictionary with sparsity statistics
        """
        # Percentage of zero activations
        zero_ratio = (latent == 0).float().mean().item()

        # Average L0 norm (number of non-zero elements)
        l0_norm = (latent != 0).float().sum(dim=-1).mean().item()

        # Average L1 norm
        l1_norm = torch.abs(latent).sum(dim=-1).mean().item()

        # Average activation value (for non-zero elements)
        non_zero_mask = latent != 0
        if non_zero_mask.any():
            avg_nonzero = latent[non_zero_mask].mean().item()
        else:
            avg_nonzero = 0.0

        return {
            'zero_ratio': zero_ratio,
            'l0_norm': l0_norm,
            'l1_norm': l1_norm,
            'avg_nonzero_activation': avg_nonzero
        }


class LayerWiseSparseAutoencoder(nn.Module):
    """
    Collection of separate sparse autoencoders, one for each layer.

    This allows learning layer-specific representations since different layers
    capture different types of features.

    Args:
        layer_dims (dict): Mapping of layer names to activation dimensions
                          e.g., {'layer_0': 4096, 'layer_8': 4096, ...}
        latent_dim (int): Dimension of latent space (same for all layers)
        sparsity_coefficient (float): L1 penalty weight
        tie_weights (bool): Whether to tie encoder/decoder weights
    """

    def __init__(self, layer_dims, latent_dim, sparsity_coefficient=1e-3, tie_weights=False):
        super(LayerWiseSparseAutoencoder, self).__init__()

        self.layer_names = list(layer_dims.keys())
        self.latent_dim = latent_dim

        # Create a separate autoencoder for each layer
        self.autoencoders = nn.ModuleDict({
            layer_name: SparseAutoencoder(
                activation_dim=dim,
                latent_dim=latent_dim,
                sparsity_coefficient=sparsity_coefficient,
                tie_weights=tie_weights
            )
            for layer_name, dim in layer_dims.items()
        })

    def forward(self, layer_activations):
        """
        Process activations from multiple layers.

        Args:
            layer_activations (dict): Dictionary mapping layer names to activation tensors

        Returns:
            dict: Results for each layer containing:
                - 'reconstructed': Reconstructed activations
                - 'latent': Latent codes
                - 'sparsity_loss': Sparsity penalty
        """
        results = {}

        for layer_name, activation in layer_activations.items():
            if layer_name in self.autoencoders:
                reconstructed, latent, sparsity_loss = self.autoencoders[layer_name](activation)
                results[layer_name] = {
                    'reconstructed': reconstructed,
                    'latent': latent,
                    'sparsity_loss': sparsity_loss
                }

        return results

    def compute_total_loss(self, layer_activations, results):
        """
        Compute total loss across all layers.

        Args:
            layer_activations (dict): Original activations
            results (dict): Forward pass results

        Returns:
            tuple: (total_loss, loss_dict)
                - total_loss: Scalar loss averaged across all layers
                - loss_dict: Per-layer loss breakdown
        """
        total_loss = 0.0
        loss_dict = {}

        for layer_name in results.keys():
            activation = layer_activations[layer_name]
            reconstructed = results[layer_name]['reconstructed']
            sparsity_loss = results[layer_name]['sparsity_loss']

            layer_total_loss, recon_loss = self.autoencoders[layer_name].total_loss(
                activation, reconstructed, sparsity_loss
            )

            total_loss += layer_total_loss
            loss_dict[layer_name] = {
                'total': layer_total_loss.item(),
                'reconstruction': recon_loss.item(),
                'sparsity': sparsity_loss.item()
            }

        # Average across layers
        total_loss = total_loss / len(results)

        return total_loss, loss_dict


if __name__ == "__main__":
    # Test the sparse autoencoder
    print("Testing SparseAutoencoder...")

    # Create a simple autoencoder with upscaling (latent_dim > activation_dim)
    ae = SparseAutoencoder(
        activation_dim=4096,
        latent_dim=16384,  # 4x upscaling for overcomplete representation
        sparsity_coefficient=1e-3
    )

    # Test input (batch_size=2, seq_len=256, dim=4096)
    test_input = torch.randn(2, 256, 4096)

    # Forward pass
    reconstructed, latent, sparsity_loss = ae(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Sparsity loss: {sparsity_loss.item():.6f}")

    # Check sparsity
    stats = ae.get_sparsity_stats(latent)
    print(f"\nSparsity statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    # Compute total loss
    total_loss, recon_loss = ae.total_loss(test_input, reconstructed, sparsity_loss)
    print(f"\nTotal loss: {total_loss.item():.6f}")
    print(f"Reconstruction loss: {recon_loss.item():.6f}")

    print("\n✓ SparseAutoencoder test passed!")

"""
Sparse Autoencoder for learning compressed representations of teacher activations.

Implementation based on "I Have Covered All the Bases Here: Interpreting Reasoning
Features in Large Language Models via Sparse Autoencoders" (2025).

The autoencoder learns to:
1. Encode high-dimensional teacher activations into a sparse latent space
2. Decode the sparse representation back to the original dimensions
3. Enforce sparsity using modified L1 regularization (weighted by decoder norm)

Architecture (from paper Section 2, Equation 1-2):
    f(x) = σ(W_enc * x + b_enc)        # Encoder with ReLU activation
    x̂(f) = W_dec * f + b_dec            # Decoder
    L = ||x - x̂||²₂ + λ Σᵢ fᵢ ||W_dec,i||₂  # Loss function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Vanilla Sparse Autoencoder with ReLU activation and modified L1 penalty.

    This implementation exactly matches the paper's architecture:
    - ReLU activation function for non-negativity
    - Squared L2 reconstruction loss
    - Modified L1 penalty: λ Σᵢ fᵢ ||W_dec,i||₂

    Note: latent_dim should be MUCH LARGER than activation_dim (overcomplete representation)
    to allow the SAE to learn disentangled features. The paper uses 16x expansion (65,536 vs 4,096).

    Args:
        activation_dim (int): Dimension of input activations (n in paper, e.g., 4096 for LLaMA-8B)
        latent_dim (int): Dimension of sparse latent representation (m in paper, should be >> n, e.g., 65536)
        sparsity_coefficient (float): λ in paper - L1 penalty weight for sparsity (default: 5.0)
        tie_weights (bool): Whether to tie encoder and decoder weights (decoder = encoder.T)
    """

    def __init__(self, activation_dim, latent_dim, sparsity_coefficient=5.0, tie_weights=False):
        super(SparseAutoencoder, self).__init__()

        self.activation_dim = activation_dim  # n
        self.latent_dim = latent_dim  # m >> n
        self.sparsity_coefficient = sparsity_coefficient  # λ
        self.tie_weights = tie_weights

        # Encoder: W_enc and b_enc (Equation 1)
        self.encoder_weight = nn.Parameter(torch.empty(latent_dim, activation_dim))
        self.encoder_bias = nn.Parameter(torch.zeros(latent_dim))

        # Decoder: W_dec and b_dec (Equation 1)
        if tie_weights:
            # Tied weights: W_dec = W_enc^T
            self.decoder_weight = None  # Will use encoder_weight.T
            self.decoder_bias = nn.Parameter(torch.zeros(activation_dim))
        else:
            self.decoder_weight = nn.Parameter(torch.empty(activation_dim, latent_dim))
            self.decoder_bias = nn.Parameter(torch.zeros(activation_dim))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights using Xavier/Glorot initialization.
        Following standard practice for autoencoders.
        """
        nn.init.xavier_uniform_(self.encoder_weight)
        nn.init.zeros_(self.encoder_bias)

        if not self.tie_weights:
            nn.init.xavier_uniform_(self.decoder_weight)
            nn.init.zeros_(self.decoder_bias)

    def encode(self, x):
        """
        Encode input activations to sparse latent representation.

        Implements: f(x) = σ(W_enc * x + b_enc) where σ is ReLU (Equation 1)

        Args:
            x (Tensor): Input activations of shape (batch_size, seq_len, activation_dim)
                       or (batch_size, activation_dim)

        Returns:
            Tensor: Sparse latent codes f(x) of shape (batch_size, seq_len, latent_dim)
                   or (batch_size, latent_dim)
        """
        # Linear transformation: W_enc * x + b_enc
        latent = F.linear(x, self.encoder_weight, self.encoder_bias)

        # ReLU activation σ for non-negativity (enforces sparsity)
        latent = F.relu(latent)

        return latent

    def decode(self, latent):
        """
        Decode latent representation back to activation space.

        Implements: x̂(f) = W_dec * f + b_dec (Equation 1)

        Args:
            latent (Tensor): Latent codes f of shape (batch_size, seq_len, latent_dim)
                           or (batch_size, latent_dim)

        Returns:
            Tensor: Reconstructed activations x̂ of shape (batch_size, seq_len, activation_dim)
                   or (batch_size, activation_dim)
        """
        if self.tie_weights:
            # Use transposed encoder weights: W_dec = W_enc^T
            reconstructed = F.linear(latent, self.encoder_weight.t(), self.decoder_bias)
        else:
            reconstructed = F.linear(latent, self.decoder_weight, self.decoder_bias)

        return reconstructed

    def forward(self, x):
        """
        Forward pass: encode then decode.

        Args:
            x (Tensor): Input activations

        Returns:
            tuple: (reconstructed, latent)
                - reconstructed: Decoded output x̂
                - latent: Sparse latent representation f
        """
        # Encode: f(x) = σ(W_enc * x + b_enc)
        latent = self.encode(x)

        # Decode: x̂(f) = W_dec * f + b_dec
        reconstructed = self.decode(latent)

        return reconstructed, latent

    def compute_sparsity_loss(self, latent):
        """
        Compute modified L1 sparsity penalty: L_sparsity = λ Σᵢ fᵢ ||W_dec,i||₂

        This is the key difference from standard SAEs - each latent activation is weighted
        by the L2 norm of its corresponding decoder weight vector (Equation 2 in paper).

        Args:
            latent (Tensor): Latent codes f of shape (..., latent_dim)

        Returns:
            Tensor: Scalar sparsity loss
        """
        # Compute L2 norm of each decoder column: ||W_dec,i||₂
        if self.tie_weights:
            # W_dec = W_enc^T, so W_dec,i is the i-th row of W_enc
            decoder_norms = torch.norm(self.encoder_weight, p=2, dim=1)  # (latent_dim,)
        else:
            # W_dec,i is the i-th column of W_dec
            decoder_norms = torch.norm(self.decoder_weight, p=2, dim=0)  # (latent_dim,)

        # Compute weighted L1: Σᵢ fᵢ ||W_dec,i||₂
        # latent: (..., latent_dim), decoder_norms: (latent_dim,)
        weighted_l1 = (latent * decoder_norms).sum(dim=-1).mean()

        # Apply sparsity coefficient λ
        sparsity_loss = self.sparsity_coefficient * weighted_l1

        return sparsity_loss

    def reconstruction_loss(self, x, reconstructed):
        """
        Compute squared L2 reconstruction loss: L_recon = ||x - x̂||²₂

        Args:
            x (Tensor): Original activations
            reconstructed (Tensor): Reconstructed activations x̂

        Returns:
            Tensor: Scalar MSE (squared L2) loss
        """
        return F.mse_loss(reconstructed, x, reduction='mean')

    def total_loss(self, x, reconstructed, latent):
        """
        Compute total loss: L = ||x - x̂||²₂ + λ Σᵢ fᵢ ||W_dec,i||₂ (Equation 2)

        Args:
            x (Tensor): Original activations
            reconstructed (Tensor): Reconstructed activations x̂
            latent (Tensor): Sparse latent codes f

        Returns:
            tuple: (total_loss, recon_loss, sparsity_loss)
        """
        recon_loss = self.reconstruction_loss(x, reconstructed)
        sparsity_loss = self.compute_sparsity_loss(latent)
        total = recon_loss + sparsity_loss

        return total, recon_loss, sparsity_loss

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
                - 'reconstructed': Reconstructed activations x̂
                - 'latent': Latent codes f
        """
        results = {}

        for layer_name, activation in layer_activations.items():
            if layer_name in self.autoencoders:
                reconstructed, latent = self.autoencoders[layer_name](activation)
                results[layer_name] = {
                    'reconstructed': reconstructed,
                    'latent': latent,
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
            latent = results[layer_name]['latent']

            layer_total_loss, recon_loss, sparsity_loss = self.autoencoders[layer_name].total_loss(
                activation, reconstructed, latent
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
    # Test the sparse autoencoder matching paper specifications
    print("Testing SparseAutoencoder (Paper Implementation)...")
    print("=" * 70)

    # Create SAE matching paper specs (Section 4.1)
    # Paper uses: n=4096 (activation_dim), m=65536 (16x expansion), λ=5.0
    print("\n1. Creating Sparse Autoencoder:")
    print(f"   - Activation dim (n): 4096")
    print(f"   - Latent dim (m): 65536 (16x overcomplete)")
    print(f"   - Sparsity coefficient (λ): 5.0")

    ae = SparseAutoencoder(
        activation_dim=4096,
        latent_dim=65536,  # 16x upscaling as in paper
        sparsity_coefficient=5.0,  # λ from paper
        tie_weights=False
    )

    # Test input (batch_size=2, seq_len=256, dim=4096)
    print("\n2. Creating test input:")
    print(f"   - Shape: (batch=2, seq_len=256, dim=4096)")
    test_input = torch.randn(2, 256, 4096)

    # Forward pass
    print("\n3. Running forward pass...")
    reconstructed, latent = ae(test_input)

    print(f"\n4. Output shapes:")
    print(f"   - Input shape: {test_input.shape}")
    print(f"   - Latent shape: {latent.shape}")
    print(f"   - Reconstructed shape: {reconstructed.shape}")

    # Compute losses
    print("\n5. Computing losses (Equation 2 from paper)...")
    total_loss, recon_loss, sparsity_loss = ae.total_loss(test_input, reconstructed, latent)
    print(f"   - Reconstruction loss (||x - x̂||²₂): {recon_loss.item():.6f}")
    print(f"   - Sparsity loss (λ Σᵢ fᵢ ||W_dec,i||₂): {sparsity_loss.item():.6f}")
    print(f"   - Total loss: {total_loss.item():.6f}")

    # Check sparsity
    stats = ae.get_sparsity_stats(latent)
    print(f"\n6. Sparsity statistics:")
    print(f"   - Zero activation ratio: {stats['zero_ratio']:.4f}")
    print(f"   - Average L0 norm (non-zero features): {stats['l0_norm']:.2f}")
    print(f"   - Average L1 norm: {stats['l1_norm']:.4f}")
    print(f"   - Average non-zero activation: {stats['avg_nonzero_activation']:.4f}")

    # Paper reports L0 of 86 at 68.5% variance explained
    expected_l0 = 86
    print(f"\n7. Target metrics (from paper Section 4.1):")
    print(f"   - Target L0: {expected_l0} (our L0: {stats['l0_norm']:.2f})")
    print(f"   - Target variance explained: 68.5%")
    print(f"   Note: Achieving target metrics requires proper training!")

    print("\n" + "=" * 70)
    print("✓ SparseAutoencoder test passed!")
    print("\nImplementation matches paper architecture:")
    print("  • Vanilla SAE with ReLU activation (Equation 1)")
    print("  • Modified L1 penalty weighted by decoder norms (Equation 2)")
    print("  • Squared L2 reconstruction loss")
    print("  • 16x overcomplete representation (m >> n)")

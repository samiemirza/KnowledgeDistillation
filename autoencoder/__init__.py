"""
Sparse Autoencoder module for learning compressed representations of teacher activations.
"""

from .sparse_autoencoder import SparseAutoencoder, LayerWiseSparseAutoencoder
from . import config

__all__ = [
    'SparseAutoencoder',
    'LayerWiseSparseAutoencoder',
    'config'
]

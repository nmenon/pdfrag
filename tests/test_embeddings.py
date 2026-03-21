# ABOUTME: Tests for EmbeddingGenerator, including device configuration.
# ABOUTME: Verifies that embeddings run on the specified device (CPU or CUDA).

import pytest
from pdfrag.embeddings import EmbeddingGenerator


def test_embedding_generator_accepts_device_param():
    """EmbeddingGenerator must accept a device parameter and run on CPU."""
    gen = EmbeddingGenerator(device="cpu")
    assert gen.model.device.type == "cpu"

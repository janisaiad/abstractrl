"""Smoke GPU : zéro téléchargement ; ignoré si CUDA indisponible."""

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA requis")
def test_cuda_matmul_small():
  a = torch.randn(256, 256, device="cuda", dtype=torch.float32)
  b = torch.randn(256, 256, device="cuda", dtype=torch.float32)
  c = a @ b
  assert c.shape == (256, 256)
  assert c.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA requis")
def test_torch_scatter_import():
  pytest.importorskip("torch_scatter")

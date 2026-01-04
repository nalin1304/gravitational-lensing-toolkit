"""
Test PINN model with adaptive pooling for variable input sizes.
Tests Step 6 of todo list.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.pinn import PhysicsInformedNN


@pytest.fixture
def pinn_model():
    """Create PINN model instance."""
    model = PhysicsInformedNN(
        input_size=64,
        dropout_rate=0.2
    )
    model.eval()
    return model


def test_pinn_accepts_64x64(pinn_model):
    """Test PINN with standard 64x64 input."""
    batch_size = 4
    x = torch.randn(batch_size, 1, 64, 64)
    
    with torch.no_grad():
        params, logits = pinn_model(x)
    
    assert params.shape == (batch_size, 5), f"Expected (4, 5), got {params.shape}"
    assert logits.shape == (batch_size, 3), f"Expected (4, 3), got {logits.shape}"
    print(f"✅ 64x64 input: params={params.shape}, logits={logits.shape}")


def test_pinn_accepts_128x128(pinn_model):
    """Test PINN with 128x128 input (larger than training size)."""
    batch_size = 2
    x = torch.randn(batch_size, 1, 128, 128)
    
    with torch.no_grad():
        params, logits = pinn_model(x)
    
    assert params.shape == (batch_size, 5), f"Expected (2, 5), got {params.shape}"
    assert logits.shape == (batch_size, 3), f"Expected (2, 3), got {logits.shape}"
    print(f"✅ 128x128 input: params={params.shape}, logits={logits.shape}")


def test_pinn_accepts_256x256(pinn_model):
    """Test PINN with 256x256 input (much larger than training size)."""
    batch_size = 1
    x = torch.randn(batch_size, 1, 256, 256)
    
    with torch.no_grad():
        params, logits = pinn_model(x)
    
    assert params.shape == (batch_size, 5), f"Expected (1, 5), got {params.shape}"
    assert logits.shape == (batch_size, 3), f"Expected (1, 3), got {logits.shape}"
    print(f"✅ 256x256 input: params={params.shape}, logits={logits.shape}")


def test_pinn_batch_consistency(pinn_model):
    """Test that different batch sizes produce consistent output shapes."""
    input_sizes = [64, 128, 256]
    batch_sizes = [1, 2, 4]
    
    results = []
    
    for size in input_sizes:
        for batch in batch_sizes:
            x = torch.randn(batch, 1, size, size)
            with torch.no_grad():
                params, logits = pinn_model(x)
            
            assert params.shape == (batch, 5), f"Inconsistent params shape at size={size}, batch={batch}"
            assert logits.shape == (batch, 3), f"Inconsistent logits shape at size={size}, batch={batch}"
            results.append((size, batch, params.shape, logits.shape))
    
    print(f"\n✅ All {len(results)} size/batch combinations passed:")
    for size, batch, p_shape, l_shape in results:
        print(f"   {size}x{size}, batch={batch}: params={p_shape}, logits={l_shape}")


def test_pinn_output_range(pinn_model):
    """Test that model outputs are in reasonable ranges."""
    x = torch.randn(4, 1, 128, 128)
    
    with torch.no_grad():
        params, logits = pinn_model(x)
    
    # Parameters should be finite
    assert torch.isfinite(params).all(), "Parameters contain NaN or Inf"
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"
    
    # Logits should have some variation (not all zeros)
    assert logits.std() > 0.001, "Logits have no variation"
    
    print(f"✅ Output ranges valid: params std={params.std():.4f}, logits std={logits.std():.4f}")


if __name__ == "__main__":
    print("Testing PINN Adaptive Pooling Implementation (Step 6)")
    print("=" * 60)
    
    model = PhysicsInformedNN(input_size=64, dropout_rate=0.2)
    model.eval()
    
    print("\nTest 1: 64x64 input")
    test_pinn_accepts_64x64(model)
    
    print("\nTest 2: 128x128 input")
    test_pinn_accepts_128x128(model)
    
    print("\nTest 3: 256x256 input")
    test_pinn_accepts_256x256(model)
    
    print("\nTest 4: Batch consistency")
    test_pinn_batch_consistency(model)
    
    print("\nTest 5: Output ranges")
    test_pinn_output_range(model)
    
    print("\n" + "=" * 60)
    print("✅ All PINN adaptive pooling tests PASSED!")
    print("Step 6 complete - ready for Step 7 (NFW deflection)")

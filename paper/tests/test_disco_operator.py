import pytest
import torch
import numpy as np
import sys
import os

# Add src directory to path to import disco_operator
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.operators.disco_operator import DiscoOperator, DiscoEncoder, disco_operator_base, disco_operator_large


class TestDiscoEncoder:
    """Test suite for DiscoEncoder component"""
    
    def test_encoder_forward_basic(self):
        """Test basic forward pass of DiscoEncoder"""
        encoder = DiscoEncoder(
            grid_size=64, 
            patch_size=8, 
            in_chans=1, 
            history=4,
            embed_dim=128, 
            depth=4, 
            num_heads=8
        )
        
        batch_size = 2
        x = torch.randn(batch_size, 1, 64, 4)  # (batch, channels, height, time)
        
        output = encoder(x)
        
        # Check output shape: (batch, num_patches + 1, embed_dim)
        expected_patches = 64 // 8  # 8 patches
        assert output.shape == (batch_size, expected_patches + 1, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_encoder_different_configs(self):
        """Test encoder with different configurations"""
        configs = [
            {"grid_size": 32, "patch_size": 4, "in_chans": 2, "history": 8, "embed_dim": 64},
            {"grid_size": 128, "patch_size": 16, "in_chans": 3, "history": 16, "embed_dim": 256},
        ]
        
        for config in configs:
            encoder = DiscoEncoder(**config, depth=2, num_heads=4)
            batch_size = 1
            x = torch.randn(batch_size, config["in_chans"], config["grid_size"], config["history"])
            
            output = encoder(x)
            expected_patches = config["grid_size"] // config["patch_size"]
            assert output.shape == (batch_size, expected_patches + 1, config["embed_dim"])


class TestDiscoOperator:
    """Test suite for DiscoOperator"""
    
    def test_disco_operator_forward_basic(self):
        """Test basic forward pass of DiscoOperator"""
        model = DiscoOperator(
            grid_size=64,
            patch_size=8,
            in_chans=1,
            history=4,
            embed_dim=128,
            encoder_depth=2,
            num_heads=8,
            time_future=1,
            n_output_scalar_components=1,
            hidden_channels=32
        )
        
        batch_size = 2
        x = torch.randn(batch_size, 1, 64, 4)  # (batch, channels, height, time_history)
        
        output = model(x)
        
        # Check output shape: (batch, output_channels, height, time_future)
        assert output.shape == (batch_size, 1, 64, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_disco_operator_different_outputs(self):
        """Test DiscoOperator with different output configurations"""
        model = DiscoOperator(
            grid_size=32,
            patch_size=4,
            in_chans=2,
            history=8,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            time_future=3,
            n_output_scalar_components=2,
            n_output_vector_components=1,  # This adds 2 more channels (vector has 2 components in 1D)
            hidden_channels=16
        )
        
        batch_size = 1
        x = torch.randn(batch_size, 2, 32, 8)
        
        output = model(x)
        
        # Expected output channels: 2 scalar + 1*2 vector = 4 total
        expected_output_channels = 2 + 1 * 2  # n_output_scalar + n_output_vector * 2
        assert output.shape == (batch_size, expected_output_channels, 32, 3)
    
    def test_disco_operator_gradient_flow(self):
        """Test that gradients flow through the model properly"""
        model = DiscoOperator(
            grid_size=32,
            patch_size=4,
            in_chans=1,
            history=4,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            time_future=1,
            n_output_scalar_components=1,
            hidden_channels=16
        )
        
        x = torch.randn(1, 1, 32, 4, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist and are non-zero for at least some parameters
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_grad = True
                break
        
        assert has_grad, "No gradients found in model parameters"
        assert x.grad is not None, "Input gradients not computed"
    
    def test_disco_operator_batch_consistency(self):
        """Test that model produces consistent outputs for different batch sizes"""
        model = DiscoOperator(
            grid_size=32,
            patch_size=4,
            in_chans=1,
            history=4,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            time_future=1,
            n_output_scalar_components=1,
            hidden_channels=16
        )
        
        # Same input, different batch sizes
        x_single = torch.randn(1, 1, 32, 4)
        x_batch = x_single.repeat(3, 1, 1, 1)
        
        model.eval()  # Ensure deterministic behavior
        with torch.no_grad():
            output_single = model(x_single)
            output_batch = model(x_batch)
        
        # Check that the first element of batch output matches single output
        assert torch.allclose(output_single, output_batch[:1], atol=1e-6)


class TestDiscoOperatorPresets:
    """Test suite for preset DiscoOperator configurations"""
    
    def test_disco_operator_base(self):
        """Test base preset configuration"""
        model = disco_operator_base(
            grid_size=64,
            patch_size=8,
            in_chans=1,
            history=4,
            time_future=1
        )
        
        x = torch.randn(1, 1, 64, 4)
        output = model(x)
        assert output.shape == (1, 1, 64, 1)
    
    def test_disco_operator_large(self):
        """Test large preset configuration"""
        model = disco_operator_large(
            grid_size=32,  # Smaller for faster testing
            patch_size=4,
            in_chans=1,
            history=4,
            time_future=1
        )
        
        x = torch.randn(1, 1, 32, 4)
        output = model(x)
        assert output.shape == (1, 1, 32, 1)


class TestDiscoOperatorEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_invalid_grid_patch_ratio(self):
        """Test that invalid grid_size/patch_size ratios are handled"""
        # This should work fine as long as grid_size is divisible by patch_size
        with pytest.raises(AssertionError):
            model = DiscoOperator(
                grid_size=65,  # Not divisible by patch_size=8
                patch_size=8,
                in_chans=1,
                history=4
            )
            x = torch.randn(1, 1, 65, 4)
            model(x)
    
    def test_mismatched_input_dimensions(self):
        """Test error handling for mismatched input dimensions"""
        model = DiscoOperator(
            grid_size=32,
            patch_size=4,
            in_chans=2,  # Expects 2 channels
            history=4,
            embed_dim=64,
            encoder_depth=2
        )
        
        # Wrong number of channels
        x_wrong_channels = torch.randn(1, 1, 32, 4)  # Only 1 channel instead of 2
        with pytest.raises(AssertionError):
            model(x_wrong_channels)
        
        # Wrong spatial size
        x_wrong_spatial = torch.randn(1, 2, 16, 4)  # 16 instead of 32
        with pytest.raises(AssertionError):
            model(x_wrong_spatial)
        
        # Wrong temporal size
        x_wrong_temporal = torch.randn(1, 2, 32, 8)  # 8 instead of 4
        with pytest.raises(AssertionError):
            model(x_wrong_temporal)


def test_memory_efficiency():
    """Test that the model doesn't consume excessive memory"""
    model = DiscoOperator(
        grid_size=64,
        patch_size=8,
        in_chans=1,
        history=4,
        embed_dim=128,
        encoder_depth=4,
        num_heads=8,
        time_future=1,
        hidden_channels=32
    )
    
    # Test with reasonably sized batch
    batch_size = 4
    x = torch.randn(batch_size, 1, 64, 4)
    
    # Should not raise memory errors
    output = model(x)
    assert output.shape == (batch_size, 1, 64, 1)


if __name__ == "__main__":
    # Run basic tests when script is executed directly
    print("Running basic DiscoOperator tests...")
    
    # Test encoder
    print("Testing DiscoEncoder...")
    test_encoder = TestDiscoEncoder()
    test_encoder.test_encoder_forward_basic()
    print("✓ DiscoEncoder basic test passed")
    
    # Test operator
    print("Testing DiscoOperator...")
    test_operator = TestDiscoOperator()
    test_operator.test_disco_operator_forward_basic()
    print("✓ DiscoOperator basic test passed")
    
    test_operator.test_disco_operator_gradient_flow()
    print("✓ DiscoOperator gradient flow test passed")
    
    # Test presets
    print("Testing preset configurations...")
    test_presets = TestDiscoOperatorPresets()
    test_presets.test_disco_operator_base()
    print("✓ Base preset test passed")
    
    print("All basic tests passed! ✓")
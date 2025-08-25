#!/usr/bin/env python3

import torch
from mae1d import MaskedAutoencoderViT

def test_mae1d_forward():
    """Test forward pass with a very small MAE 1D model"""
    
    # Create a tiny model for testing
    model = MaskedAutoencoderViT(
        grid_size=64,      # Small spatial dimension
        patch_size=8,      # Small patches
        in_chans=2,        # Few channels
        history=4,         # Short history
        embed_dim=32,      # Small embedding
        depth=2,           # Few encoder layers
        num_heads=4,       # Few attention heads
        decoder_embed_dim=16,  # Small decoder embedding
        decoder_depth=1,   # Single decoder layer
        decoder_num_heads=2,   # Few decoder heads
        mlp_ratio=2.0      # Small MLP ratio
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create test input: (batch_size, channels, spatial, temporal)
    batch_size = 2
    x = torch.randn(batch_size, 2, 64, 4)
    
    print(f"Input shape: {x.shape}")
    print(f"Expected patches: {64 // 8} = {model.patch_embed.num_patches}")
    
    # Forward pass
    try:
        loss, pred, mask = model(x, mask_ratio=0.5)
        
        print(f"Forward pass successful!")
        print(f"Loss: {loss.item():.4f}")
        print(f"Prediction shape: {pred.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask ratio (actual): {mask.float().mean().item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_mae1d_forward()
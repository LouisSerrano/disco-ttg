#!/usr/bin/env python3

def test_simple_import():
    """Test basic import only"""
    try:
        from disco_operator import DiscoOperator
        print("✓ Successfully imported DiscoOperator")
        
        # Test basic instantiation  
        model = DiscoOperator(
            grid_size=64,  # smaller for testing
            patch_size=8,
            history=4,
            embed_dim=128,
            encoder_depth=2,
            hidden_channels=32
        )
        print("✓ Successfully created DiscoOperator instance")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing basic disco_operator import...")
    success = test_simple_import()
    if success:
        print("\n🎉 Import test passed!")
    else:
        print("\n❌ Import test failed!")
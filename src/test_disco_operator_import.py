#!/usr/bin/env python3

import torch
import torch.nn as nn

def test_disco_operator_import():
    """Test that disco_operator can be imported and instantiated"""
    try:
        from disco_operator import DiscoOperator, disco_operator_mini
        print("✓ Successfully imported DiscoOperator")
        
        # Test basic instantiation
        #model = DiscoOperator()
        #print("✓ Successfully created DiscoOperator instance")
        
        # Test convenience function
        model = disco_operator_mini()
        print("✓ Successfully created disco_operator_base instance")
        
        # Test forward pass with dummy data
        batch_size = 2
        channels = 3
        height = 256
        time_history = 16
        
        # Create dummy input
        x = torch.randn(batch_size, channels, height, time_history)
        print(f"✓ Created dummy input with shape: {x.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
            print(f"✓ Forward pass successful! Output shape: {output.shape}")
            
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing disco_operator with new oned_unet import...")
    success = test_disco_operator_import() 
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")
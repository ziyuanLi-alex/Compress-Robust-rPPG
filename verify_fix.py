
import torch
import sys
import os

# Ensure we can import the project modules
sys.path.append(os.getcwd())

try:
    print("Importing mamba_ssm...")
    import mamba_ssm
    print("mamba_ssm imported.")
    
    from mamba_ssm import Mamba
    print("Mamba class imported.")

    print("Initializing Mamba model...")
    model = Mamba(
        d_model=16,
        d_state=16,
        d_conv=4,
        expand=2
    ).cuda()
    print("Mamba model initialized.")

    print("Running forward pass...")
    x = torch.randn(1, 10, 16).cuda()
    out = model(x)
    print(f"Forward pass successful. Output shape: {out.shape}")
    
    # Check backward pass
    print("Running backward pass...")
    loss = out.sum()
    loss.backward()
    print("Backward pass successful.")

except Exception as e:
    print(f"Verification FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

"""
Assignment 3 - Neural Network Reliability Verification
Marabou Example Script
Author: <YOUR NAME>
Date: November 2025

This script demonstrates:
    1. Loading an external model and dataset (not included in Marabou resources)
    2. Converting a PyTorch model into ONNX
    3. Running the ONNX model through Marabou
    4. Verifying a simple property
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import onnx
import os
from maraboupy import Marabou, MarabouCore

from maraboupy.MarabouNetworkONNX import MarabouNetworkONNX

# Try to import onnx-simplifier if available
try:
    import onnxsim
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False
    print("[INFO] onnx-simplifier not available, skipping model simplification")

# ----------------------------------------------------------
# 1. DEFINE AN EXTERNAL MODEL (small MLP suitable for Marabou)
# ----------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Use Flatten in Sequential to see if it helps
        self.layers = nn.Sequential(
            nn.Flatten(start_dim=1),  # Flatten from dimension 1 onwards
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)


# ----------------------------------------------------------
# 2. LOAD EXTERNAL DATASET (MNIST, not in Marabou resources)
# ----------------------------------------------------------
def load_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    return loader


# ----------------------------------------------------------
# 3. TRAIN (OPTIONAL) OR LOAD MODEL, THEN EXPORT TO ONNX
# ----------------------------------------------------------
def export_model_to_onnx(model, onnx_path="model.onnx"):
    dummy_input = torch.randn(1, 1, 28, 28)
    temp_path = onnx_path + ".temp"
    
    # Export to ONNX with opset 11
    torch.onnx.export(
        model,
        dummy_input,
        temp_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True
    )
    print(f"[INFO] Exported ONNX model to {temp_path}")
    
    # Try to simplify the model if onnxsim is available
    if ONNXSIM_AVAILABLE:
        try:
            print("[INFO] Simplifying ONNX model...")
            model_onnx = onnx.load(temp_path)
            model_simplified, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_simplified, onnx_path)
                print(f"[INFO] Simplified model saved to {onnx_path}")
                os.remove(temp_path)
            else:
                print("[WARNING] Model simplification check failed, using original")
                os.rename(temp_path, onnx_path)
        except Exception as e:
            print(f"[WARNING] Model simplification failed: {e}, using original")
            os.rename(temp_path, onnx_path)
    else:
        os.rename(temp_path, onnx_path)
        print(f"[INFO] Model saved to {onnx_path}")


# ----------------------------------------------------------
# 4. RUN MARABOU VERIFICATION
# ----------------------------------------------------------
def run_marabou(onnx_path="model.onnx"):
    print("[INFO] Loading ONNX model into Marabou...")
    # Try using MarabouNetworkONNX directly as an alternative to Marabou.read_onnx()
    try:
        network = MarabouNetworkONNX(onnx_path)
    except Exception as e:
        print(f"[WARNING] Direct MarabouNetworkONNX failed: {e}")
        print("[INFO] Trying Marabou.read_onnx() instead...")
        network = Marabou.read_onnx(onnx_path)

    # Input and output variable indices
    input_vars = network.inputVars[0].flatten()
    output_vars = network.outputVars[0].flatten()

    # ------------------------------------------------------
    # Example property:
    #   For some input x in [0, 1]^784
    #   Verify: output[3] >= output[5]
    #
    # This is arbitrary — you must define a property in your report.
    # ------------------------------------------------------

    print("[INFO] Setting input bounds...")
    for i in input_vars:
        network.setLowerBound(i, 0.0)
        network.setUpperBound(i, 1.0)

    print("[INFO] Adding verification constraint: output[3] >= output[5]")
    eq = MarabouCore.Equation(MarabouCore.Equation.GE)

    eq.addAddend(1, output_vars[3])
    eq.addAddend(-1, output_vars[5])
    eq.setScalar(0)
    network.addEquation(eq)

    # ------------------------------------------------------
    # Solve
    # ------------------------------------------------------
    print("[INFO] Running Marabou solver...")
    exitCode, vals, stats = network.solve()

    if exitCode == "sat":
        print("[RESULT] Property is SAT (counterexample found).")
        print("Example violating input:", [vals[v] for v in input_vars[:10]], "...")
        print("Output values:")
        for i, out_var in enumerate(output_vars):
            print(f"  output {i} = {vals[out_var]}")
    else:
        print("[RESULT] Property is UNSAT (model satisfies the constraint).")
        print(f"Exit code: {exitCode}")

    print("[STATS]", stats)


# ----------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------
if __name__ == "__main__":
    print("\n========== Assignment 3: Marabou Verification ==========\n")

    # Step 1: Define model
    model = SimpleMLP()

    # Step 2: Load dataset (optional use)
    dataset = load_dataset()

    # Step 3: Export to ONNX
    export_model_to_onnx(model, "mnist_mlp.onnx")

    # Step 4: Run Marabou
    run_marabou("mnist_mlp.onnx")

    print("\n===================== DONE =====================\n")

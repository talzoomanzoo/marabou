"""
Test script for Assignment 3:
Runs the entire pipeline:
- Load dataset
- Build model
- Export ONNX
- Verify with Marabou
"""

from marabou_model import (
    SimpleMLP,
    load_dataset,
    export_model_to_onnx,
    run_marabou
)

if __name__ == "__main__":
    print("\n========== Assignment 3: Marabou Verification Test ==========\n")

    model = SimpleMLP()
    ds = load_dataset()

    export_model_to_onnx(model, "mnist_mlp.onnx")

    exitCode, vals, stats = run_marabou("mnist_mlp.onnx")

    if exitCode == "sat":
        print("\n[RESULT] SAT (counterexample found)")
    else:
        print("\n[RESULT] UNSAT (property holds)")

    print("\n[STATS]:")
    print(stats)

    print("\n===================== DONE =====================\n")

"""
Main implementation code for Assignment 3:
- Defines model
- Loads dataset
- Exports ONNX
- Prepares Marabou verification
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import onnx
import os

from maraboupy import MarabouCore
from maraboupy.MarabouNetworkONNX import MarabouNetworkONNX

# Optional ONNX simplifier
try:
    import onnxsim
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False


# -------------------------
# Model
# -------------------------
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)


# -------------------------
# Dataset
# -------------------------
def load_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return DataLoader(ds, batch_size=1, shuffle=False)


# -------------------------
# ONNX Export
# -------------------------
def export_model_to_onnx(model, filename="mnist_mlp.onnx"):
    dummy = torch.randn(1, 1, 28, 28)
    temp_file = filename + ".temp"

    torch.onnx.export(
        model, dummy, temp_file,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True
    )

    if ONNXSIM_AVAILABLE:
        model_onnx = onnx.load(temp_file)
        simplified, ok = onnxsim.simplify(model_onnx)
        if ok:
            onnx.save(simplified, filename)
            os.remove(temp_file)
            return
    os.rename(temp_file, filename)


# -------------------------
# Marabou Verification
# -------------------------
def run_marabou(onnx_path):
    try:
        network = MarabouNetworkONNX(onnx_path)
    except:
        from maraboupy import Marabou
        network = Marabou.read_onnx(onnx_path)

    input_vars = network.inputVars[0].flatten()
    output_vars = network.outputVars[0].flatten()

    # Bounds
    for v in input_vars:
        network.setLowerBound(v, 0)
        network.setUpperBound(v, 1)

    # Property: output[3] >= output[5]
    eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    eq.addAddend(1, output_vars[3])
    eq.addAddend(-1, output_vars[5])
    eq.setScalar(0)
    network.addEquation(eq)

    return network.solve()

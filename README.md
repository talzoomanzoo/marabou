# Assignment 3 — Neural Network Reliability Verification

**Course:** Neural Network Verification  
**Due Date:** November 28, 2025  
**Author:** <YOUR NAME>

---

## 📌 Overview

This assignment introduces the use of **Marabou**, a constraint-based neural network verification tool. The objectives are:

1. Explore Marabou's `resources/` directory
2. Load and verify a neural network model that is **not** included in Marabou's repository
3. Use an external dataset
4. Export a model to ONNX and run Marabou verification
5. Document the process and verification results

This README describes the setup, model/dataset choices, execution instructions, verification property, and encountered issues.

---

## 📁 File Structure

```
Assignment3/
│
├── marabou_model.py        # Main implementation (model, dataset, ONNX export, verification)
├── test_marabou.py         # Test script to run the entire pipeline
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── report.pdf              # Summary report (provided by student)
├── data/                   # MNIST dataset (auto-downloaded)
│   └── MNIST/
└── mnist_mlp.onnx          # ONNX model (auto-generated)
```

---

## 🧰 Environment Setup

### Python Version

Python 3.10 or later is recommended.

### Required Packages

Create a `requirements.txt` with:

```
torch
torchvision
onnx
maraboupy
numpy
onnxscript
onnxruntime
```

Install with:

```bash
pip install -r requirements.txt
```

### Installing Marabou

To install Marabou:

```bash
git clone https://github.com/NeuralNetworkVerification/Marabou
cd Marabou
pip install maraboupy
```

**Note:** Warnings about TensorFlow absence can be ignored, as this assignment uses the ONNX parser.

---

## 🔬 External Model & Dataset

### Custom Model

A small PyTorch MLP used as the verified network:

- **Input size:** 28×28
- **Hidden layer:** 64 units + ReLU
- **Output:** 10 logits

This model is not included in Marabou's resource directory.

### Dataset

The MNIST test set, downloaded automatically using:

```python
torchvision.datasets.MNIST
```

This fulfills the assignment requirement of using an external dataset.

---

## 📤 ONNX Export Notes

PyTorch exports models using opset 18 by default. Even when opset 11 is requested, PyTorch attempts (and fails) to downgrade:

```
Failed to convert the model to the target version 11...
```

This is expected and harmless. Marabou successfully loads ONNX models at opset 18 for simple feed-forward networks.

---

## 🚀 How to Run the Script

To run the entire pipeline:

```bash
python test_marabou.py
```

The script will:

1. Load MNIST dataset
2. Build the custom MLP model
3. Export the model to ONNX
4. Load the ONNX model into Marabou
5. Apply a verification constraint
6. Run the solver
7. Print SAT/UNSAT and solver statistics

### Code Organization

- **`marabou_model.py`**: Contains the main implementation:
  - `SimpleMLP`: PyTorch model definition
  - `load_dataset()`: MNIST dataset loader
  - `export_model_to_onnx()`: ONNX export function
  - `run_marabou()`: Marabou verification function

- **`test_marabou.py`**: Test script that orchestrates the entire workflow

---

## ✅ Verification Property

The verification property used is:

**output[3] ≥ output[5]**

for all inputs satisfying:

**0 ≤ input[i] ≤ 1**

### Interpretation:

- **UNSAT** → The property holds for all inputs
- **SAT** → A counterexample exists where output[3] < output[5]

If SAT, the violating input assignment is shown.

---

## 🔧 Troubleshooting Notes

### 1. ONNX Conversion Warning

PyTorch cannot convert opset 18 → 11. This is expected and does not affect verification. Use the generated opset 18 model; Marabou accepts it.

### 2. Missing MarabouNetworkONNX Import

Some installations require explicitly importing:

```python
from maraboupy.MarabouNetworkONNX import MarabouNetworkONNX
```

### 3. Equation Type Errors

Your Marabou version does not support:
- `eq.type = ...`
- `eq.setType(...)`

**Correct usage:**

```python
eq = MarabouCore.Equation(MarabouCore.Equation.GE)
```

Note: Do NOT use `.value` (e.g., `MarabouCore.Equation.GE.value`) as it will cause a TypeError. Pass the enum member directly.

### 4. Solve Return Value

The `solve()` method returns three values:

```python
exitCode, vals, stats = network.solve()
```

Make sure to unpack all three values, not just two.

---

## 📊 Outputs

The script prints:

- **SAT or UNSAT** result
- **Counterexample input** (if SAT)
- **Output values** for all output variables (if SAT)
- **Marabou solver statistics**

These results are discussed in the accompanying report.pdf.

---

## 📚 Assignment Source

This work follows the instructions from the uploaded Assignment 3 PDF.

---

## 👤 Author

**<Minju Gwak>**  
Neural Network Verification — Fall 2025

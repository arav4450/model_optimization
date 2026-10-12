# Deep Learning Model Optimization with ONNX

An implementation of deep learning model optimization techniques using **ONNX** and **ONNX Runtime**, designed for **CPU-only execution on 16GB RAM**.


---

## Repository Structure

```text
model_optimization/
│
├── 01_onnx_simplifier_and_runtime.py   
├── 02_model_quantization.py            
├── 03_model_pruning.py                 
├── 04_knowledge_distillation.py        
├── main.py                             # Unified CLI runner for all modules
│
├── exported_models/                    # Output directory for exported & optimized ONNX models
├── pyproject.toml                      # Project metadata and dependencies
└── README.md                           # Documentation and tutorial guide
```

---

## Optimization Concepts & Implementations

### 1. ONNX Simplifier and ONNX Runtime (`01_onnx_simplifier_and_runtime.py`)

* **Core Principles**:
  * **Static vs. Dynamic Shape Export**: Exporting PyTorch graphs with fixed shapes vs. dynamic axes (`batch_size`).
  * **ONNX Simplifier (`onnxsim`)**: Folds constant tensors at compile-time, eliminates redundant reshape/identity nodes, and propagates shapes across the graph.
  * **ONNX Runtime Graph Optimization Levels**:
    * `ORT_DISABLE_ALL`: Executes graph without node transformation.
    * `ORT_ENABLE_BASIC`: Constant folding and redundant node elimination.
    * `ORT_ENABLE_EXTENDED`: Layer/operator fusions (e.g., `Conv + Relu`, `Gemm + Relu`).
    * `ORT_ENABLE_ALL`: Complete graph optimizations including CPU layout transforms (`NCHWc`).
  * **Serialization**: Saving the optimized graph via `sess_options.optimized_model_filepath`.
* **Results**:
  * Original graph: **15 nodes**
  * Simplified graph: **6 nodes (60.0% reduction)**
  * ORT Optimized graph: **5 nodes (66.7% reduction)**
  * Max numerical difference: **< 1e-6** 

---

### 2. Model Quantization (`02_model_quantization.py`)

* **Core Principles**:
  * **Quantization Formula**: Maps 32-bit floating point ($\text{FP32}$) to signed 8-bit integers ($\text{INT8}$):
    $$q = \text{round}\left(\frac{x}{S}\right) + Z \quad \Longleftrightarrow \quad x \approx S \cdot (q - Z)$$
    where $S$ is the scale factor and $Z$ is the zero-point.
  * **Preprocessing (`quant_pre_process`)**: Propagates shape inference and prepares operators prior to quantization.
  * **Dynamic Quantization (`quantize_dynamic`)**:
    * Weights are converted to INT8 offline (`QuantType.QInt8`).
    * Activations remain in float and are dynamically quantized during execution.
    * **No calibration data required**; ideal for memory-bound networks (Transformers, LSTMs, MLPs).
  * **Static Quantization (`quantize_static`)**:
    * Both weights and activations are quantized to INT8 ahead of time using QDQ operators (`QuantizeLinear` / `DequantizeLinear`).
    * Implements a custom `CalibrationDataReader` (`get_next()`, `rewind()`) to sample data distributions.
* **Results**:
  * Baseline FP32 model size: **654.1 KB**
  * Dynamic INT8 model size: **170.9 KB (3.83x compression / 73.9% size reduction)**
  * Static INT8 model size: **171.2 KB (3.82x compression / 73.8% size reduction)**
  * Output cosine similarity: **> 0.9999** (negligible precision loss).

---

### 3. Model Pruning (`03_model_pruning.py`)

* **Core Principles**:
  * **Sparsity Metric**:
    $$\text{Sparsity} = \frac{\sum \text{zeros}}{\sum \text{total weights}} \times 100\%$$
  * **Unstructured Pruning**: Sets individual weights below an L1 magnitude threshold to zero using `torch.nn.utils.prune.l1_unstructured`. The matrix dimensions remain unchanged.
  * **Structured Channel Pruning (Cascading Channels)**:
    * Physically prunes lowest-norm output filters from `Conv1`.
    * **Cascading update**: The input channel dimension of the subsequent `Conv2` layer must shrink proportionally to match `Conv1`'s new channel count.
    * Directly reduces computation (FLOPs) and physical parameter count on standard CPU hardware without specialized sparse kernels.
* **Results**:
  * Baseline Conv1 weight shape: `[32, 3, 3, 3]`, Conv2: `[64, 32, 3, 3]` (**20,138 parameters**)
  * Unstructured pruned: `[32, 3, 3, 3]` with **50.0% zeroed weights**
  * Structured pruned: Conv1 weight shrank to `[24, 3, 3, 3]`, Conv2 shrank to `[64, 24, 3, 3]` (**15,298 parameters, 24.0% reduction**)
  * ONNX model size: **81.2 KB $\rightarrow$ 61.9 KB**.

---

### 4. Knowledge Distillation (`04_knowledge_distillation.py`)

* **Core Principles**:
  * **Teacher-Student Setup**: A high-capacity Teacher guides a lightweight Student model.
  * **Distillation Loss (Hinton formulation)**:
    $$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{soft}} + (1 - \alpha) \cdot \mathcal{L}_{\text{hard}}$$
    * **Soft Target Loss** (KL Divergence with Temperature $T$):
      $$\mathcal{L}_{\text{soft}} = \text{KLDiv}\left(\log\text{softmax}\left(\frac{z_s}{T}\right), \text{softmax}\left(\frac{z_t}{T}\right)\right) \times T^2$$
      The $T^2$ scaling compensates for the gradient magnitude of softened softmax.
    * **Hard Target Loss**: Standard Cross-Entropy between student predictions and ground truth:
      $$\mathcal{L}_{\text{hard}} = \text{CrossEntropy}(z_s, y)$$
* **Results**:
  * Teacher Network: **27,139 parameters**, **107.1 KB**, **96.00% accuracy**
  * Student Network: **643 parameters (97.6% smaller)**, **3.0 KB**
  * Standalone Student Accuracy: **95.33%**
  * Distilled Student Accuracy: **96.00%** (matches teacher accuracy)
  * ONNX Runtime CPU Latency: **1.27x speedup**.

---

## How to Run

### Environment Setup
Activate the environment or install dependencies:
```bash
# Using uv (recommended)
uv run python main.py --all

# Or activate virtual environment
.venv\Scripts\activate
python main.py --all
```

### Running Individual Scripts
Each script is completely standalone and can be executed independently:

```bash
# 1. Graph simplification and ORT optimization
python 01_onnx_simplifier_and_runtime.py

# 2. Dynamic and Static INT8 Quantization
python 02_model_quantization.py

# 3. Unstructured and Structured Channel Pruning
python 03_model_pruning.py

# 4. Knowledge Distillation & Edge ONNX Export
python 04_knowledge_distillation.py
```

Or using the CLI orchestrator:
```bash
python main.py --module 1   # Run Module 1
python main.py --module 2   # Run Module 2
python main.py --module 3   # Run Module 3
python main.py --module 4   # Run Module 4
```

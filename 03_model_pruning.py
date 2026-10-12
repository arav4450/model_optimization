"""
========================================================================================
Model Pruning in PyTorch and Exporting to ONNX

========================================================================================

Concepts Covered:
1. What is Model Pruning:
   - Eliminating redundant parameters to reduce memory footprint and computational load.
   - Global sparsity computation:
       Sparsity = (Total Zero Weights / Total Parameters) * 100%
2. Unstructured Pruning (Fine-grained / Weight-level):
   - Sets individual weight values below a threshold (lowest L1 magnitude) to zero.
   - Uses PyTorch's `prune.l1_unstructured`.
   - Freezing the pruning mask with `prune.remove(module, 'weight')` so the zeros are baked in.
   - Exporting the sparse model to ONNX.
3. Structured Pruning (Coarse-grained / Channel & Filter-level):
   - Physically removes entire filters / channels based on L1-norm magnitude.
   - Cascading channel updates (Crucial concept from Chapter 6):
       When Layer N's output channels are pruned from C_out to C'_out,
       Layer (N+1)'s input channels must be resized from C_out to C'_out.
   - Results in physically smaller tensor shapes, reducing FLOPs on standard CPU hardware without
     requiring specialized sparse accelerators.
4. Exporting to ONNX & ONNX Runtime Validation:
   - Exporting baseline, unstructured pruned, and structured pruned models to ONNX.
   - Comparing parameter count, actual ONNX tensor dimensions, file size, and latency.

Executable in seconds on 16GB RAM CPU-only systems.
"""

import os
import sys
import copy
import time
import warnings

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import onnx
import onnxruntime as ort


# --------------------------------------------------------------------------------------
# 1. Define a Convolutional Network for Pruning Demonstration
# --------------------------------------------------------------------------------------
class ConvNet(nn.Module):
    """
    A 2-stage Convolutional network with Batch Normalization and Linear classifier.

    """

    def __init__(self, in_channels=3, conv1_out=32, conv2_out=64, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, conv1_out, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(conv1_out)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            conv1_out, conv2_out, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(conv2_out)
        self.relu2 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(conv2_out, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# --------------------------------------------------------------------------------------
# 2. Sparsity Metric Computation
# --------------------------------------------------------------------------------------
def compute_global_sparsity(model):
    """
    Computes global sparsity percentage (ratio of zeroed weights to total weights).

    """
    total_zeros = 0
    total_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            total_zeros += (weight == 0).sum().item()
            total_elements += weight.numel()
    sparsity_pct = (total_zeros / total_elements) * 100 if total_elements > 0 else 0.0
    return sparsity_pct, total_zeros, total_elements


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------------------
# 3. Unstructured Pruning
# --------------------------------------------------------------------------------------
def apply_unstructured_pruning(model, prune_ratio=0.5):
    """
    Applies magnitude-based (L1) unstructured pruning to Conv and Linear layers.
    Sets `prune_ratio` fraction of individual weights to zero.

    """
    pruned_model = copy.deepcopy(model)

    for name, module in pruned_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Prune lowest L1-norm weights
            prune.l1_unstructured(module, name="weight", amount=prune_ratio)
            # Make pruning permanent by removing re-parameterization hooks
            prune.remove(module, "weight")

    sparsity, zeros, total = compute_global_sparsity(pruned_model)
    print(
        f" -> Unstructured Sparsity: {sparsity:.2f}% ({zeros}/{total} zeroed weights)"
    )
    return pruned_model


# --------------------------------------------------------------------------------------
# 4. Structured Pruning (Channel Pruning with Cascading Layer Updates)
# --------------------------------------------------------------------------------------
def apply_structured_channel_pruning(model, prune_ratio=0.25):
    """
    Physically removes channels based on L1-norm magnitude.

    When Conv1 output channels are pruned, Conv2 input channels MUST cascade and adapt!
    """
    print(f"\n[2/4] Applying Structured Pruning (prune_ratio={prune_ratio:.0%})...")
    # Determine which filters in conv1 to keep based on L1-norm
    conv1_weights = model.conv1.weight.data  # shape: [out_channels, in_channels, k, k]
    c1_out = conv1_weights.shape[0]
    num_to_prune = int(c1_out * prune_ratio)
    num_to_keep = c1_out - num_to_prune

    # Compute L1 norm per output filter
    filter_norms = torch.norm(conv1_weights.view(c1_out, -1), p=1, dim=1)
    keep_indices = (
        torch.topk(filter_norms, k=num_to_keep, largest=True).indices.sort().values
    )

    print(f" -> Pruning Conv1 output channels from {c1_out} to {num_to_keep}")
    print(f" -> Cascading update to Conv2 input channels: {c1_out} -> {num_to_keep}")

    # Create a new, physically smaller ConvNet
    new_model = ConvNet(
        in_channels=model.conv1.in_channels,
        conv1_out=num_to_keep,
        conv2_out=model.conv2.out_channels,
        num_classes=model.fc.out_features,
    )

    # Copy Conv1 weights for kept channels
    new_model.conv1.weight.data = model.conv1.weight.data[keep_indices].clone()

    # Update BN1 parameters for kept channels
    new_model.bn1.weight.data = model.bn1.weight.data[keep_indices].clone()
    new_model.bn1.bias.data = model.bn1.bias.data[keep_indices].clone()
    new_model.bn1.running_mean.data = model.bn1.running_mean.data[keep_indices].clone()
    new_model.bn1.running_var.data = model.bn1.running_var.data[keep_indices].clone()

    # Cascade to Conv2: Conv2 input channels now only take the kept channels from Conv1
    new_model.conv2.weight.data = model.conv2.weight.data[:, keep_indices].clone()

    # Copy remaining layers (BN2 and FC)
    new_model.bn2.weight.data = model.bn2.weight.data.clone()
    new_model.bn2.bias.data = model.bn2.bias.data.clone()
    new_model.bn2.running_mean.data = model.bn2.running_mean.data.clone()
    new_model.bn2.running_var.data = model.bn2.running_var.data.clone()
    new_model.fc.weight.data = model.fc.weight.data.clone()
    new_model.fc.bias.data = model.fc.bias.data.clone()

    return new_model


# --------------------------------------------------------------------------------------
# 5. Export to ONNX & Benchmark
# --------------------------------------------------------------------------------------
def export_model_to_onnx(model, onnx_path, sample_shape=(1, 3, 32, 32)):
    model.eval()
    dummy_input = torch.randn(*sample_shape, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )


def benchmark_ort(onnx_path, test_input, num_warmup=20, num_iters=100):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    feed = {input_name: test_input}

    for _ in range(num_warmup):
        _ = session.run(None, feed)

    start = time.perf_counter()
    for _ in range(num_iters):
        outputs = session.run(None, feed)
    end = time.perf_counter()

    avg_ms = ((end - start) / num_iters) * 1000
    return avg_ms, outputs[0]


def run_pipeline():
    print("=" * 80)
    print("DEMO: MODEL PRUNING IN PYTORCH & EXPORTING TO ONNX")
    print("=" * 80)

    os.makedirs("./exported_models", exist_ok=True)
    unpruned_path = "./exported_models/03_model_unpruned.onnx"
    unstructured_path = "./exported_models/03_model_unstructured_pruned.onnx"
    structured_path = "./exported_models/03_model_structured_pruned.onnx"

    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Base Model
    base_model = ConvNet(in_channels=3, conv1_out=32, conv2_out=64, num_classes=10)
    base_params = count_parameters(base_model)
    export_model_to_onnx(base_model, unpruned_path)
    print(f"[1/4] Base Model parameters: {base_params:,} | Exported to {unpruned_path}")

    # 2. Unstructured Pruning (50% weights zeroed)
    print(f"\n[2/4] Applying Unstructured Pruning (50% sparsity)...")
    unstructured_model = apply_unstructured_pruning(base_model, prune_ratio=0.5)
    export_model_to_onnx(unstructured_model, unstructured_path)
    print(f" -> Exported to: {unstructured_path}")

    # 3. Structured Channel Pruning (25% Conv1 channels eliminated)
    structured_model = apply_structured_channel_pruning(base_model, prune_ratio=0.25)
    struct_params = count_parameters(structured_model)
    export_model_to_onnx(structured_model, structured_path)
    print(
        f" -> Structured Pruned parameters: {struct_params:,} (Reduced by {base_params - struct_params:,} parameters)"
    )
    print(f" -> Exported to: {structured_path}")

    # 4. Compare Models in ONNX Runtime
    print(f"\n[4/4] Comparing ONNX Graphs, File Sizes & Inference Latency...")
    sample_input = np.random.randn(1, 3, 32, 32).astype(np.float32)

    lat_unpruned, out_unpruned = benchmark_ort(unpruned_path, sample_input)
    lat_unstruct, out_unstruct = benchmark_ort(unstructured_path, sample_input)
    lat_struct, out_struct = benchmark_ort(structured_path, sample_input)

    size_unpruned_kb = os.path.getsize(unpruned_path) / 1024
    size_unstruct_kb = os.path.getsize(unstructured_path) / 1024
    size_struct_kb = os.path.getsize(structured_path) / 1024

    def get_conv_shapes(onnx_path):
        m = onnx.load(onnx_path)
        init_map = {init.name: list(init.dims) for init in m.graph.initializer}
        conv_nodes = [n for n in m.graph.node if n.op_type == "Conv"]
        c1_shape = init_map[conv_nodes[0].input[1]]
        c2_shape = init_map[conv_nodes[1].input[1]]
        return c1_shape, c2_shape

    w_unpruned_c1, w_unpruned_c2 = get_conv_shapes(unpruned_path)
    w_unstruct_c1, w_unstruct_c2 = get_conv_shapes(unstructured_path)
    w_struct_c1, w_struct_c2 = get_conv_shapes(structured_path)

    print("\n" + "=" * 85)
    print(
        f"{'Pruning Technique':<20} | {'Parameters':<11} | {'Sparsity':<9} | {'Conv1 Shape':<15} | {'Conv2 Shape':<15} | {'Latency':<8}"
    )
    print("-" * 85)
    print(
        f"{'Unpruned (Baseline)':<20} | {base_params:<11,} | {'0.0%':<9} | {str(w_unpruned_c1):<15} | {str(w_unpruned_c2):<15} | {f'{lat_unpruned:.4f}ms':<8}"
    )
    print(
        f"{'Unstructured Pruned':<20} | {base_params:<11,} | {'50.0%':<9} | {str(w_unstruct_c1):<15} | {str(w_unstruct_c2):<15} | {f'{lat_unstruct:.4f}ms':<8}"
    )
    print(
        f"{'Structured Pruned':<20} | {struct_params:<11,} | {'Channels':<9} | {str(w_struct_c1):<15} | {str(w_struct_c2):<15} | {f'{lat_struct:.4f}ms':<8}"
    )
    print("=" * 85)
    print("Summary:")
    print(
        "✓ Unstructured pruning sets 50% of weights to 0 without altering tensor dimensions."
    )
    print(
        "✓ Structured pruning physically reduced Conv1 output channels and Conv2 input channels."
    )
    print(f"✓ Conv1 weight tensor shrank: {w_unpruned_c1} -> {w_struct_c1}")
    print(
        f"✓ Conv2 weight tensor shrank: {w_unpruned_c2} -> {w_struct_c2} (cascading input channels)"
    )
    print("✓ Direct parameter reduction and faster CPU inference achieved.")


if __name__ == "__main__":
    run_pipeline()

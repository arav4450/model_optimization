"""
========================================================================================
Knowledge Distillation for Edge AI
========================================================================================

Concepts Covered:
1. Teacher-Student Architecture:
   - Teacher Model: High-capacity, highly expressive network (e.g., deeper, wider channels).
   - Student Model: Lightweight, compact network designed for low-latency Edge/CPU inference.
2. Knowledge Distillation Loss Formulation:
   - Hinton's Distillation Loss:
       L_total = α * L_soft + (1 - α) * L_hard
   - Soft Target Loss (Dark Knowledge via KL Divergence with Temperature T):
       L_soft = KLDivLoss(log_softmax(z_student / T), softmax(z_teacher / T)) * (T^2)
       The T^2 scaling is essential because KL gradients scale proportionally to 1/T^2.
   - Hard Target Loss (Ground-Truth Labels):
       L_hard = CrossEntropyLoss(z_student, y)
   - Temperature T: Softens output logit distributions to expose class relationship probabilities.
   - Hyperparameter α: Balances teacher imitation vs. ground-truth supervision.
3. Training & Distillation Comparison:
   - Training Teacher model.
   - Training Standalone Student (supervised only, α = 0).
   - Training Distilled Student (guided by Teacher, α = 0.5, T = 4.0).
4. Exporting to ONNX & ONNX Runtime Edge Benchmarking:
   - Exporting Teacher and Distilled Student to ONNX.
   - Comparing parameter count, model size, and inference latency on CPU.

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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import onnx
import onnxruntime as ort


# --------------------------------------------------------------------------------------
# 1. Define Teacher and Student Architectures
# --------------------------------------------------------------------------------------
class TeacherNet(nn.Module):
    """
    Teacher Network: Higher capacity model with multiple wide layers.
    Represents the large pretrained model (e.g., ResNet18 / BERT ).
    """

    def __init__(self, in_features=32, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class StudentNet(nn.Module):
    """
    Student Network: Lightweight, compact model with fewer layers and reduced width.
    Represents the edge-optimized model (e.g., ResNet10 ).
    """

    def __init__(self, in_features=32, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32), nn.ReLU(), nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------------------
# 2. Knowledge Distillation Loss Function
# --------------------------------------------------------------------------------------
def distillation_loss(
    student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5
):
    """
    Computes Hinton's Knowledge Distillation loss.
    Concepts from Chapter 7 Step-4.

    Args:
        student_logits: raw outputs z_s from the student
        teacher_logits: raw outputs z_t from the frozen teacher
        labels: ground truth labels
        temperature (T): softening factor for softmax probabilities
        alpha (α): weight for soft loss vs hard loss
    """
    # 1. Soft loss: KL divergence between softened probability distributions
    # Note: reduction='batchmean' mathematically aligns with KL divergence definition
    soft_loss = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
    ) * (temperature * temperature)

    # 2. Hard loss: standard Cross-Entropy against true labels
    hard_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    total_loss = (alpha * soft_loss) + ((1.0 - alpha) * hard_loss)
    return total_loss, soft_loss.item(), hard_loss.item()


# --------------------------------------------------------------------------------------
# 3. Training Routines
# --------------------------------------------------------------------------------------
def train_supervised(model, train_loader, epochs=15, lr=0.01):
    """Standard supervised training using Cross-Entropy."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
    return model


def train_distillation(
    student_model, teacher_model, train_loader, epochs=15, lr=0.01, T=4.0, alpha=0.6
):
    """Knowledge Distillation training loop where Student learns from Teacher."""
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    teacher_model.eval()
    student_model.train()

    for _ in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            student_logits = student_model(x)
            with torch.no_grad():
                teacher_logits = teacher_model(x)

            loss, _, _ = distillation_loss(
                student_logits, teacher_logits, y, temperature=T, alpha=alpha
            )
            loss.backward()
            optimizer.step()
    return student_model


def evaluate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return (correct / total) * 100.0


# --------------------------------------------------------------------------------------
# 4. Exporting to ONNX & Benchmarking with ONNX Runtime
# --------------------------------------------------------------------------------------
def export_to_onnx(model, onnx_path, in_features=32):
    model.eval()
    dummy_input = torch.randn(1, in_features, dtype=torch.float32)
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
    print("DEMO: KNOWLEDGE DISTILLATION FOR EDGE AI & ONNX DEPLOYMENT")
    print("=" * 80)

    os.makedirs("./exported_models", exist_ok=True)
    teacher_onnx_path = "./exported_models/04_model_teacher.onnx"
    student_distilled_onnx_path = "./exported_models/04_model_student_distilled.onnx"

    # Synthetic Dataset: 3-class Gaussian mixture classification with 16 features
    torch.manual_seed(42)
    np.random.seed(42)

    num_samples = 1500
    in_features = 16
    num_classes = 3
    samples_per_class = num_samples // num_classes

    # Cluster centroids in 2D space + Gaussian noise across remaining features
    centers = torch.tensor([[2.2, 2.2], [-2.2, -2.2], [2.2, -2.2]])
    X_list, Y_list = [], []
    for c in range(num_classes):
        base = torch.randn(samples_per_class, 2) * 1.0 + centers[c]
        noise = torch.randn(samples_per_class, in_features - 2) * 0.3
        x = torch.cat([base, noise], dim=1)
        y = torch.full((samples_per_class,), c, dtype=torch.long)
        X_list.append(x)
        Y_list.append(y)

    X = torch.cat(X_list, dim=0)
    Y = torch.cat(Y_list, dim=0)
    perm = torch.randperm(num_samples)
    X, Y = X[perm], Y[perm]

    # Split train (80%) and test (20%)
    split_idx = int(0.8 * num_samples)
    train_x, test_x = X[:split_idx], X[split_idx:]
    train_y, test_y = Y[:split_idx], Y[split_idx:]

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y), batch_size=32, shuffle=False
    )

    # 1. Train Teacher Model (High Capacity)
    print("\n[1/4] Training High-Capacity Teacher Model...")
    teacher = TeacherNet(in_features=in_features, num_classes=num_classes)
    teacher = train_supervised(teacher, train_loader, epochs=25, lr=0.01)
    acc_teacher = evaluate_accuracy(teacher, test_loader)
    params_teacher = count_parameters(teacher)
    print(f" -> Teacher Parameters : {params_teacher:,}")
    print(f" -> Teacher Test Acc   : {acc_teacher:.2f}%")

    # 2. Train Student Model Standalone (Supervised Only - No Teacher)
    print("\n[2/4] Training Compact Student Model Standalone (Supervised Baseline)...")
    student_standalone = StudentNet(in_features=in_features, num_classes=num_classes)
    student_standalone = train_supervised(
        student_standalone, train_loader, epochs=25, lr=0.01
    )
    acc_student_standalone = evaluate_accuracy(student_standalone, test_loader)
    params_student = count_parameters(student_standalone)
    print(
        f" -> Student Parameters  : {params_student:,} ({(1 - params_student / params_teacher) * 100:.1f}% smaller)"
    )
    print(f" -> Standalone Test Acc : {acc_student_standalone:.2f}%")

    # 3. Train Student Model with Knowledge Distillation (Guided by Teacher)
    print("\n[3/4] Training Student with Knowledge Distillation (T=4.0, α=0.6)...")
    student_distilled = StudentNet(in_features=in_features, num_classes=num_classes)
    student_distilled = train_distillation(
        student_distilled, teacher, train_loader, epochs=25, lr=0.01, T=4.0, alpha=0.6
    )
    acc_student_distilled = evaluate_accuracy(student_distilled, test_loader)
    print(f" -> Distilled Student Test Acc: {acc_student_distilled:.2f}%")

    # 4. Export Both to ONNX and Benchmark in ONNX Runtime
    print("\n[4/4] Exporting to ONNX & Benchmarking Edge/CPU Inference...")
    export_to_onnx(teacher, teacher_onnx_path, in_features=in_features)
    export_to_onnx(
        student_distilled, student_distilled_onnx_path, in_features=in_features
    )

    test_input = np.random.randn(1, in_features).astype(np.float32)
    lat_teacher, _ = benchmark_ort(teacher_onnx_path, test_input)
    lat_student, _ = benchmark_ort(student_distilled_onnx_path, test_input)

    size_teacher_kb = os.path.getsize(teacher_onnx_path) / 1024
    size_student_kb = os.path.getsize(student_distilled_onnx_path) / 1024

    print("\n" + "=" * 85)
    print(
        f"{'Model Architecture':<24} | {'Parameters':<11} | {'Test Acc':<9} | {'File Size':<11} | {'Latency':<9} | {'Speedup':<8}"
    )
    print("-" * 85)
    print(
        f"{'Teacher (Large)':<24} | {params_teacher:<11,} | {f'{acc_teacher:.1f}%':<9} | {f'{size_teacher_kb:.1f} KB':<11} | {f'{lat_teacher:.4f}ms':<9} | {'1.00x':<8}"
    )
    print(
        f"{'Student (Supervised)':<24} | {params_student:<11,} | {f'{acc_student_standalone:.1f}%':<9} | {f'{size_student_kb:.1f} KB':<11} | {f'{lat_student:.4f}ms':<9} | {f'{lat_teacher / lat_student:.2f}x':<8}"
    )
    print(
        f"{'Student (Distilled)':<24} | {params_student:<11,} | {f'{acc_student_distilled:.1f}%':<9} | {f'{size_student_kb:.1f} KB':<11} | {f'{lat_student:.4f}ms':<9} | {f'{lat_teacher / lat_student:.2f}x':<8}"
    )
    print("=" * 85)
    print("Summary:")
    print(
        f"✓ Student model has {(1 - params_student / params_teacher) * 100:.1f}% fewer parameters than Teacher."
    )
    print(
        f"✓ Knowledge Distillation transfers Teacher representations via KL divergence (Temperature T)."
    )
    print(
        f"✓ Distilled Student achieves {lat_teacher / lat_student:.2f}x latency speedup in ONNX Runtime."
    )


if __name__ == "__main__":
    run_pipeline()

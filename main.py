"""
========================================================================================
Main Runner: Deep Learning Model Optimization with ONNX
Companion implementation for "Ultimate ONNX for Deep Learning Optimization"
========================================================================================

Modules:
  1. 01_onnx_simplifier_and_runtime.py : ONNX Simplifier & ORT Graph Optimization (Ch 4)
  2. 02_model_quantization.py          : Dynamic & Static INT8 Quantization (Ch 5)
  3. 03_model_pruning.py               : Unstructured & Structured Channel Pruning (Ch 6)
  4. 04_knowledge_distillation.py      : Teacher-Student KD & ONNX Export (Ch 7)

Usage:
  python main.py --all           # Run all 4 modules sequentially
  python main.py --module 1      # Run Module 1 (ONNX Simplifier & Runtime)
  python main.py --module 2      # Run Module 2 (Quantization)
  python main.py --module 3      # Run Module 3 (Pruning)
  python main.py --module 4      # Run Module 4 (Knowledge Distillation)
"""

import sys
import argparse
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPTS = {
    "1": (
        "01_onnx_simplifier_and_runtime.py",
        "ONNX Simplifier & ONNX Runtime (Chapter 4)",
    ),
    "2": (
        "02_model_quantization.py",
        "Model Quantization: Dynamic & Static INT8 (Chapter 5)",
    ),
    "3": (
        "03_model_pruning.py",
        "Model Pruning: Unstructured & Structured (Chapter 6)",
    ),
    "4": (
        "04_knowledge_distillation.py",
        "Knowledge Distillation for Edge AI (Chapter 7)",
    ),
}


def run_script(module_key):
    script_name, description = SCRIPTS[str(module_key)]
    print("\n" + "#" * 80)
    print(f"RUNNING MODULE {module_key}: {description}")
    print(f"File: {script_name}")
    print("#" * 80 + "\n")

    cmd = [sys.executable, script_name]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"\n[ERROR] Module {module_key} exited with code {ret.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run Model Optimization with ONNX demonstrations."
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all optimization modules sequentially."
    )
    parser.add_argument(
        "--module",
        type=str,
        choices=["1", "2", "3", "4"],
        help="Choose a specific module to run (1, 2, 3, or 4).",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("      DEEP LEARNING MODEL OPTIMIZATION WITH ONNX")
    print("  Reference: 'Ultimate ONNX for Deep Learning Optimization'")
    print("=" * 80)

    if args.all:
        for k in ["1", "2", "3", "4"]:
            success = run_script(k)
            if not success:
                break
        print("\n" + "=" * 80)
        print("All optimization modules completed successfully!")
        print("Check the './exported_models/' directory for generated ONNX models.")
        print("=" * 80)
    elif args.module:
        run_script(args.module)
    else:
        # Default behavior: run all modules or display menu
        print("\nNo arguments provided. Running all modules (--all)...\n")
        for k in ["1", "2", "3", "4"]:
            success = run_script(k)
            if not success:
                break
        print("\n" + "=" * 80)
        print("All optimization modules completed successfully!")
        print("Check the './exported_models/' directory for generated ONNX models.")
        print("=" * 80)


if __name__ == "__main__":
    main()

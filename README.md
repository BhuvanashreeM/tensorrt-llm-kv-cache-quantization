# KV-Cache Quantization for Efficient LLM Inference

This repository contains a self-contained notebook that demonstrates how to install, build inference engines, and evaluate Large Language Models (LLMs) using TensorRT-LLM. The focus is on optimizing inference performance through techniques such as precision scaling and KV-Cache quantization.

It is intended for students and practitioners with limited compute resources who want a practical, reproducible introduction to LLM inference optimization. All experiments were conducted on an NVIDIA A100 GPU in Google Colab, making the workflow accessible without requiring direct access to on-premise datacenter hardware.

## Project Highlights

- **Colab-Ready**: The notebook is self-contained and can be run directly in Google Colab without complex local setup.
- **Student-Friendly Model Size**: Uses a 1B parameter model, large enough to show meaningful tradeoffs but small enough to run within resource-constrained environments.
- **End-to-End Demonstration**: Covers loading, quantization, and inference with multiple backends.
- **Optimization Techniques**: Demonstrates FP16 precision, INT8 KV-Cache quantization, and AWQ (activation-aware weight quantization).
- **Evaluation Across Context Lengths**: Benchmarks short-context and long-context scenarios to highlight where optimizations provide the most benefit.

## Optimization Techniques

### FP16 Precision
Reduces memory usage compared to FP32 and improves throughput, with negligible accuracy degradation.

### INT8 KV-Cache Quantization
The KV-Cache grows linearly with sequence length, becoming the main memory bottleneck in long-context inference. Quantizing it to INT8 reduces memory by 2× compared to FP16, enabling longer contexts and faster inference.

### AWQ (Activation-Aware Weight Quantization)
Quantizes weights to 4-bit or 8-bit while retaining higher precision for activations. Balances performance, memory, and accuracy, enabling efficient deployment in both datacenter and edge settings.

## Why This Matters

Long-context inference is increasingly important for applications such as code assistants, multi-document summarization, and conversational AI. Without optimization, KV-Cache memory quickly becomes a limiting factor. This project illustrates the importance of cache quantization by comparing short-context and long-context scenarios.

## Notes and Caveats

- **Model Choice**: A 1B parameter model was selected for reproducibility. Larger models can be substituted if additional GPU memory is available.
- **GPU Support**: Works on A100 and newer GPUs. Not supported on T4 GPUs, as TensorRT-LLM kernels (e.g., multi-head attention) are not optimized for them.
- **Installation**: Designed to run in Colab. For production-scale development, follow [NVIDIA's TensorRT-LLM Installation Instructions](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation.md).

## Skills Demonstrated

- Analysis of Transformer inference bottlenecks such as the KV-Cache.
- Evaluation of optimizations across both short-context and long-context workloads.
- Application of quantization methods (FP16, INT8, AWQ) to reduce memory and latency.
- Hands-on use of TensorRT-LLM in a reproducible environment.
- Communication of tradeoffs between accuracy, memory, and performance.

## TODO (Work in Progress)

- [x] Self-contained Colab notebook demonstrating KV-Cache quantization.
- [x] Evaluation on both short-context and long-context inputs.
- [ ] Integrate trtllm-bench based evaluation to standardize comparisons across backends.
- [ ] Add GPU profiling using Nsight Systems / Nsight Compute to analyze kernel-level performance.
- [ ] Explore additional precision strategies (e.g., FP8) for inference on H100-class GPUs.

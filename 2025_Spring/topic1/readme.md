# Final Project Topic 1: Run-Time Reconfigurable Matrix Multiplication Kernel with Mixed-Precision

## Overview

This project involves implementing a **run-time reconfigurable matrix multiplication kernel**. The kernel must support multiple data types and be optimized for resource utilization and latency. The design should address challenges in **machine learning (ML) workloads**, where training and inference require different precision formats.

## Motivation

Modern ML workloads often rely on **mixed-precision computations** for efficiency:
- **Training**: Uses higher precision for accuracy.
- **Inference**: Uses lower precision for faster computation and reduced resource usage.

For instance, the paper ["Hybrid 8-bit Floating Point (HFP8) Training and Inference for Deep Neural Networks"](https://proceedings.neurips.cc/paper_files/paper/2019/file/65fc9fb4897a89789352e211ca2d398f-Paper.pdf) proposes:
- **E4M3**: 4-bit exponent, 3-bit mantissa for forward passes.
- **E5M2**: 5-bit exponent, 2-bit mantissa for backpropagation.

Your task is to design a matrix multiplication kernel that supports these mixed-precision requirements while minimizing resource usage and latency.

---

## Project Requirements

### Input Specifications

- Two input matrices: **A (200×200)** and **B (200×200)**.
- The precision of the matrices will vary based on the scenario:
  - **Inference**: 8-bit fixed-point (`ap_fixed<8, 4>`), or 16-bit fixed-point (`ap_fixed<16, 5>`).
  - **Training**: 8-bit floating-point formats (E4M3 or E5M2) or 16-bit floating-point format (E5M10).
- **Total Configurations**: Five possibilities for each matrix:
  8-bit `ap_fixed`, 16-bit `ap_fixed`, E4M3, E5M2, and E5M10.
- For this project, **A** and **B** will have the same data type. For advanced goals, they can differ.

### Design Objectives

1. **Correctness**: 
   - Ensure the kernel produces correct results by comparing with 32-bit floating-point operations.
   - Quantify errors using metrics like maximum error, mean squared error (MSE), or error distribution.

2. **Configurable Precision Support**: 
   - The kernel must support all specified data types.

3. **Resource Sharing**: 
   - Optimize hardware resources to efficiently compute across multiple precision scenarios.

4. **Single Bitstream**: 
   - The hardware design must remain unchanged across all configurations.

5. **Latency Minimization**: 
   - Reduce latency for completing all matrix multiplications.

---

## Evaluation Metrics

Your design will be evaluated based on the following criteria:

| **Criterion**                         | **Weight** |
|---------------------------------------|------------|
| Mid-term Report                       | 5          |
| Final Report (with performance data)  | 25         |
| Source Code                           | 25         |
| Presentation                          | 5          |

---

## File Descriptions

- `matrix_generation.cpp`: Generates testing matrices. Modify only if you find bugs or want to improve it.
- `host.cpp`: Host code for preparing data, initiating HLS kernel execution, and collecting results.
- `hls_kernels.cpp`: Contains all HLS kernels for matrix multiplication.
- Input/Output Files:
  - **Reference Files**:
    - `matrix_a_float.bin`, `matrix_b_float.bin`, `matrix_c_float.bin`: Matrices in 32-bit floating-point format. 
  - **E4M3 Format**:
    - `matrix_a_E4M3.bin`, `matrix_b_E4M3.bin`: Input matrices stored in E4M3 format.
    - `matrix_c_E4M3.bin`: Output matrix in E4M3 format.
  - **Other Formats**:
    - `matrix_a_AP8_4.bin`, `matrix_a_AP16_5.bin`, and similar files for different precisions.

---

## Design Hints and Challenges

### Key Questions to Consider

- **Multiplier and Adder Design**:
  - How will you design shared multipliers and adders to support multiple precisions efficiently?
  - Avoid creating separate kernels for each precision to save resources.

- **Matrix Tiling and Scheduling**:
  - What tiling strategy will you use for large matrices?
  - How will you schedule multiply-accumulate (MAC) operations to maximize throughput?

- **Data Packing**:
  - How will you efficiently pack/unpack varying bit-width data into FPGA memory?
  - Can you fully utilize the DRAM bandwidth (up to 512-bit per cycle)?

- **Trade-offs**:
  - How will you balance resource sharing and parallel computation to minimize latency without excessive overhead?

### Implementation Strategies

1. **Naive Approach**:
   - Use separate multipliers for each data type. This is simple but inefficient in terms of resource usage and latency.

2. **Optimized Approach**:
   - Design configurable multipliers that reuse hardware across multiple scenarios to reduce redundancy and enhance parallelism.

---

## Deliverables

1. **HLS Implementation**:
   - Submit well-commented HLS design files explaining your design choices.
   - Use **HLSFactory** for submission unless publication is planned.

2. **Simulation Results**:
   - Demonstrate correctness for all matrix multiplication configurations.

3. **Performance Report**:
   - Accuracy: Compare output matrices with 32-bit floating-point results and compute MSE.
   - Latency: Measure latency for each configuration using **C/RTL Co-Simulation** or **LightningSim**.
   - Resource Utilization: Provide data post-implementation (after place-and-route), including the generated bitstream.

4. **Technical Report**:
   - Use LaTeX in IEEE double-column format. Include:
     - **Design Overview**:
       - Component-level details (e.g., multiplier design for mixed precisions).
       - Algorithm-level details (e.g., matrix tiling, scheduling, and buffering strategies).
     - **Experimental Results**:
       - Performance and resource usage.
       - Comparisons with baselines (e.g., unoptimized designs or alternative sharing strategies).
     - **Insights and Takeaways**:
       - Lessons learned and potential improvements.

---

## Stretch Goals (Optional)

1. **Support More Data Types**:
   - Add support for additional formats like 8-bit integer, 4-bit integer, 32-bit floating point, and 4-bit floating point.

2. **Different Data Types for A and B**:
   - Enable scenarios where matrix A (activations) uses E4M3 while matrix B (weights) uses int8.

3. **Trade-off Analysis**:
   - Investigate trade-offs between generality and resource efficiency.
   - Analyze whether resource-sharing or separate kernels offer better performance.

---

## Submission Instructions

- Submit all files (code, simulation results, reports) as a **zipped folder**.
- Ensure files are clearly labeled and easy to follow.

---

**Good luck with your project!**

# Final Project Topic 3: Neuro-Symbolic Kernel Optimization for Cognitive AI Systems

## Overview

In this project, you will implement **neuro** and **symbolic** kernels and optimize their resource utilization and latency on an FPGA. The design should address challenges in **neuro-symbolic AI workloads**, where heterogeneous neuro and symbolic components require efficient processing to enable cognitive intelligence.


## Motivation

Neuro-symbolic AI is an emerging compositional paradigm that fuses neural learning with symbolic reasoning to enhance the transparency, interpretability, and trustworthiness of AI. This approach mirrors human cognitive processes, where lower-level sensory inputs are processed by neural mechanisms (System 1: “thinking fast”), and higher-level cognitive functions involve symbolic reasoning and deduction (System 2: “thinking slow”). (["Towards Cognitive AI Systems: Workload and Characterization of Neuro-Symbolic AI"](https://ieeexplore.ieee.org/document/10590020))

Despite its impressive cognitive capabilities, enabling real-time and efficient neuro-symbolic AI remains a challenging problem, critical for numerous reasoning and human-AI applications. For example, the ["Neuro-Vector-Symbolic System"](https://www.nature.com/articles/s42256-023-00630-8) requires over a minute to process a single task even on TPU and desktop GPU. Analysis shows that neural and vector-symbolic circular convolution operations account for most of the latency.

Your task is to design efficient neuro (e.g., ResNet18) and symbolic (e.g., circular convolution) kernels while minimizing resource usage and latency.

---

## Project Requirements

### Neuro Kernel (ResNet18)
- **Purpose**: Processes input image patches, identifies feature attributes of compound objects, and stores the results.
- **Reference Code**: `neural/neural_resnet_18.py`
- **Input Data**: `neural/neural_input/image_X.npy` (X: 1-6)
- **Output Data**: `neural/neural_output/image_X.npy` (X: 1-6)
- **Running Command**: `python neural_resnet_18.py --model_path model_best.pth.tar --input_path ./neural_input/image_X.npy --output_ref_path ./neural_output/model_output_X.npy --device cuda:0` (please specify `X`)

### Symbolic Kernel (Circular Convolution)
- **Purpose**: Processes input probabilistic scene vectors, deduces underlying rules in the Raven Progressive Matrices (RPM) spatial-temporal reasoning task, and stores the results.
- **Reference Code**: `symbolic/symbolic_circular_conv.py`
- **Input Data**: `symbolic/symbolic_input/inputX_A.npy`, `symbolic/symbolic_input/inputX_B.npy` (X: 1-6)
- **Output Data**: `symbolic/symbolic_output/outputX_C.npy` (X: 1-6)
- **Running Command**: `python symbolic_circular_conv.py --input_A_path ./symbolic_input/inputX_A.npy --input_B_path ./symbolic_input/inputX_B.npy --output_ref_path ./symbolic_output/outputX_C.npy --device cuda:0` (please specify `X`)

### Design Objectives
- **Correctness**: Ensure the neuro and symbolic kernels produce correct results by comparing them with the provided golden outputs.
- **Latency Minimization**: Reduce latency for completing neuro and symbolic computations.

### Bonus (+10 Points)
- **Support for Additional Data Types**: Add support for formats such as 8-bit integer, 4-bit integer, 32-bit floating point, and 4-bit floating point.
- **Reconfigurable Support for Neuro and Symbolic Operations**: Design a kernel that is reconfigurable to efficiently support both convolution (neuro) and circular convolution (symbolic). Analyze whether resource-sharing or separate neuro and symbolic kernels offer better performance.

Our provided codebase is a portion of the end-to-end neuro-vector-symbolic system which can be referred here ["neuro-vector-symbolic system"](https://github.com/IBM/neuro-vector-symbolic-architectures-raven/tree/main).

---

## Design Hints
- **Optimization Techniques**: Utilize optimization techniques covered in this class, such as memory specialization (e.g., array partitioning, data reuse) and compute specialization (e.g., pipelining, loop unrolling, parallelism, dataflow, multithreading, loop tiling, loop fusion) to optimize the neuro and symbolic kernels.
- **ResNet**: The ResNet18 HLS implementation can be referred to [SkyNet] (https://github.com/TomG008/SkyNet/tree/master/FPGA/HLS).
- **Circular Convolution**: Is the vector-symbolic circular convolution kernel compute- or memory-bound? Is a systolic array architecture efficient for circular convolution? Explore ways to optimize dataflow and mapping for high-dimensional vector-based circular convolution.

---

## Evaluation Metrics

Your design will be evaluated based on the following criteria:

| **Criterion**                         | **Weight** |
|---------------------------------------|------------|
| Mid-term Report                       | 5          |
| Final Report (with performance data)  | 25         |
| Source Code                           | 25         |
| Presentation                          | 5          |
| Bonus                                 | 10         |

---

## Evaluation Criteria
- **End-to-End Latency**: Measure the total time required to process neuro and symbolic kernels, from reading input data to writing processed results back to DRAM.
- **Post-Implementation Resource Utilization**: Report FPGA resource usage, including DSP slices, BRAM, LUTs, and FFs.
- **Speedup Comparison**: Compare the latency of the FPGA implementation with the original Python implementation on a CPU. Report the speedup achieved.

---

## Deliverables

1. **HLS Implementation**:
   - Submit well-commented HLS design files explaining your design choices.
   - Use **HLSFactory** for submission unless publication is planned.

2. **Simulation Results**:
   - Verify correctness of your computation using simulation.

3. **Performance Report**:
   - **Latency**: Measure latency for neuro and symbolic kernel using **C/RTL Co-Simulation** or **LightningSim**.
   - **Resource Utilization**: Provide data post-implementation (after place-and-route), including the generated bitstream.

4. **Technical Report**:
   - Use LaTeX in IEEE double-column format. Include:
     - **Design Overview**:
       - Algorithm-level details, including any optimizations made to improve performance.
       - Hardware component details, focusing on the core processing elements.
     - **Experimental Results**:
       - Performance metrics (latency, throughput, etc.) and resource usage.
       - Comparisons with baseline Python implementation on a CPU.
     - **Insights and Takeaways**:
       - Lessons learned, challenges encountered, and potential areas for improvement.

---

## Submission Instructions

- Submit all files (code, simulation results, reports) as a **zipped folder**.
- Ensure files are clearly labeled and easy to follow.

---

**Good luck with your project!**
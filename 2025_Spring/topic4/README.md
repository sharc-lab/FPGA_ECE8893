# Final Project Topic 4: Neuro-Symbolic Kernel Optimization for Cognitive AI Systems

## Overview

In this project, you will implement **neuro** and **symbolic** kernels and optimize their resource utilization and latency on an FPGA. The design should address challenges in **neuro-symbolic AI workloads**, where heterogeneous neuro and symbolic components require efficient processing to enable cognitive intelligence.


## Motivation

Neuro-symbolic AI is an emerging compositional paradigm that fuses neural learning with symbolic reasoning to enhance the transparency, interpretability, and trustworthiness of AI. This approach mirrors human cognitive processes, where lower-level sensory inputs are processed by neural mechanisms (System 1: “thinking fast”), and higher-level cognitive functions involve symbolic reasoning and deduction (System 2: “thinking slow”). (["Towards Cognitive AI Systems: Workload and Characterization of Neuro-Symbolic AI"](https://ieeexplore.ieee.org/document/10590020))

Despite its impressive cognitive capabilities, enabling real-time and efficient neuro-symbolic AI remains a challenging problem, critical for numerous reasoning and human-AI applications. For example, the ["Neuro-Vector-Symbolic System"](https://www.nature.com/articles/s42256-023-00630-8) requires over a minute to process a single task even on TPU and desktop GPU. Analysis shows that neural and vector-symbolic circular convolution operations account for most of the latency.

Your task is to design efficient neural (e.g., one layer of ResNet18) and symbolic (e.g., circular convolution) kernels while minimizing resource usage and latency. The project builds upon the neuro-vector-symbolic system [codebase](https://github.com/IBM/neuro-vector-symbolic-architectures-raven/tree/main).

---

## Project Requirements

### Neural Kernel (One Layer of ResNet18)
- **Purpose**: Processes input image patches, extracts feature attributes of compound objects, and stores the results.
- **Reference Code**: `neural/neural_conv.py`
- **Input Data**: download inputs from [here](https://drive.google.com/file/d/1pelBT7OxsK2hBqJ2Ww-Sq9nyIl8M4FRx/view?usp=drive_link) and place the six files (`input_X.npy`, where `X` = 1-6) under `neural/neural_input/`.
- **Layer Weights**: download model weights from [here](https://drive.google.com/file/d/1R66bKtFIaP0_OqjvWlSrLZS35ePiTnMG/view?usp=sharing)
- **Output Data**: `neural/neural_output/output_X.npy` (`X` = 1-6)
- **Running Command**: `python neural_conv.py --model_path model_best.pth.tar --input_path ./neural_input/input_X.npy --output_ref_path ./neural_output/output_X.npy --device cuda:0` (Replace `X` with the corresponding file number)

### Symbolic Kernel (Circular Convolution)
- **Purpose**: Processes input probabilistic scene vectors, deduces underlying rules for the Raven Progressive Matrices (RPM) spatial-temporal reasoning task, and stores results.
- **Reference Code**: `symbolic/symbolic_circular_conv.py`
- **Input Data**: `symbolic/symbolic_input/inputX_A.npy`, `symbolic/symbolic_input/inputX_B.npy` (`X` = 1-6)
- **Output Data**: `symbolic/symbolic_output/outputX_C.npy` (`X` = 1-6)
- **Running Command**: `python symbolic_circular_conv.py --input_A_path ./symbolic_input/inputX_A.npy --input_B_path ./symbolic_input/inputX_B.npy --output_ref_path ./symbolic_output/outputX_C.npy --device cuda:0` (Replace `X` with the corresponding file number)

### Design Objectives
- **Correctness**: Ensure the neural and symbolic kernels produce correct results by comparing them with the provided golden outputs.
- **Latency Minimization**: Reduce latency for completing neural and symbolic computations.

### Bonus 1 (+10 Points)
- **Task**: Implement a complete ResNet18
- **Reference Code**: `bonus/ResNet18/neural_resnet_18.py`
- **Input Data**: `bonus/ResNet18/ResNet_input/image_X.npy` (`X` = 1-6)
- **Output Data**: `bonus/ResNet18/ResNet_output/model_output_X.npy` (`X` = 1-6)
- **Running Command**: `python neural_resnet_18.py --model_path model_best.pth.tar --input_path ./ResNet_input/image_X.npy --output_ref_path ./ResNet_output/model_output_X.npy --device cuda:0` (Replace `X` with the corresponding file number)

### Bonus 2 (+10 Points)
- **Task**: Reconfigurable Support for Neuro and Symbolic Operations. Design a kernel that is reconfigurable to efficiently support both convolution kernel (neural) and circular convolution kenrel (symbolic). Analyze whether resource-sharing or separate neural and symbolic kernels offer better performance.

---

## Design Hints
- **Neural Input Pre-processing**: Convert `.npy` data to `.bin` format using `utility/convert_between_npy_bin.py`, and load `.bin` in C format using `utility/load_binary_in_C.cpp`. The `input_1.bin` example used in `utility/load_binary_in_C.cpp` can be downloaded from [here](https://drive.google.com/file/d/1I_MmbxUvrWPNFv1uQHOj1g2O5_11CtCT/view?usp=sharing).
- **ResNet Layer Implementation**: to improve hardware efficiency, you can fuse convolution and batch normalization operations, please refer example functions in `utility/fuse_conv_bn.py`.
- **ResNet Model Implementation**: The complete ResNet18 HLS implementation can be referred to ["SkyNet"] (https://github.com/TomG008/SkyNet/tree/master/FPGA/HLS).
- **Optimization Techniques**: Utilize optimization techniques covered in this class, such as memory specialization (e.g., array partitioning, data reuse) and compute specialization (e.g., pipelining, loop unrolling, parallelism, dataflow, multithreading, loop tiling, loop fusion) to optimize the neural and symbolic kernels.
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
| Bonus 1                               | 10         |
| Bonus 2                               | 10         |

---

## Evaluation Criteria
- **Latency**: Measure the total time required to process neuro and symbolic kernels, from reading input data to writing processed results back to DRAM.
- **Post-Implementation Resource Utilization**: Report FPGA resource usage, including DSP slices, BRAM, LUTs, and FFs.

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

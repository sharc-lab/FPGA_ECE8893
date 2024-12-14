# Lab 1 - Introduction to HLS

## Overview

In this lab, we will cover the fundamentals of High-Level Synthesis (HLS) design using a simple example: element-wise vector multiplication. The following techniques will be explored:

- Using Vitis HLS through command-line interfaces.
- Understanding the HLS design process, including:
  - **C Compilation:** Translate high-level C code into an executable.
  - **C Simulation:** Verify the functionality of the C code.
  - **C Synthesis:** Generate RTL from C code.
  - **C/RTL Co-Simulation:** Perform cycle-accurate RTL simulation.
  - **RTL implementation:** Perform place and route and generate final bitstream that can be used to program FPGA.
  - Reading and interpreting HLS synthesis reports, with a focus on latency and resource usage.

---

## How to Run

1. **C Compilation:** Run `make`.
2. **HLS Synthesis:** Run `vitis_hls -f script.tcl`.
   - Review the `script.tcl` file to understand each command. In the file, it includes steps:
     - C synthesis
     - C/RTL co-simulation
     - RTL place and route, which will take much longer time
   - Upon successful execution, you can find the reports in the following locations:
     - **C-Synthesis Report:** `project_1/solution1/syn/report`.
     - **C/RTL Co-Simulation Report:** `project_1/solution1/sim/report`.
     - **Post-implementation Results:** `project_1/solution1/impl/report/verilog/export_impl.rpt`.

---

## Questions and Exploration

Please explore the following questions in this lab:

### 1. **Report Design Performance**
- Correctly report the HLS design's performance and resource usage.
- The resource usage must be post-implementation: this ensures that your design can be successfully implemented and mapped to FPGA.

### 2. **Performance Study**
- Try to use `#pragma HLS array_partition` and `#pragma HLS unroll` to improve performance by reducing computation latency.

  **Questions:**
  - Which arrays are you partitioning?
  - Are the arrays stored as DRAM or BRAM?
  - Can you partition an array stored in DRAM? If not, what should you do?

### 3. **Data Type Study**
- Analyze the effects of changing the data type (e.g., floating-point or double).
  - How does the change impact latency and resource usage?

### 4. **Scalability Study** (optional)
- Examine the scalability of the design by increasing the array size from 100 to 10,000,000.

  **Questions:**
  - Does the technique used in the performance study still apply?
  - If not, what alternative approach would you use?

### 5. **Takeaways**
- Summarize your insights and findings from these studies.

---

## Notes and Submission Instructions

- Report latency in two forms:
  - From the HLS synthesized report.
  - From C/RTL co-simulation.
- This is an **individual assignment.** You may discuss with classmates but copying is not allowed.
- Submit a technical report discussing all the above questions.
- The report should ideally use LaTeX and follow the IEEE double-column format.

---

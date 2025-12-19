# Lab 2 – Memory-Centric Optimization with Fixed-Point Stencil

## Overview

In this lab, we will study **memory-centric optimization** using a **2D multi-point stencil kernel** implemented with **fixed-point arithmetic**. Compared to Lab 1, this lab shifts the focus from simple loop-level optimizations to **data reuse, memory bandwidth, and compute pattern restructuring**.

The target kernel performs a **weighted 2D stencil computation** over multiple time steps. This type of computation is widely used in scientific simulations, image processing, and numerical solvers, and is often **memory-bound** rather than compute-bound.

The following techniques will be explored:

- Understanding memory access patterns in nested loops.
- Reusing data through buffering and loop restructuring.
- Applying HLS optimizations such as:
  - loop pipelining  
  - array partitioning  
  - selective unrolling  
  - loop reordering or rewriting  
- Interpreting HLS reports with emphasis on **memory bottlenecks**, **latency**, and **resource usage**.

---

## Exploration Tips

Consider exploring the following questions in this lab:

### 1. **Observe Design Change**
- Constantly monitoring design correctness and performance change as you make optimizations.

### 2. **Memory Reuse and Bandwidth Study**
- Identify the memory access pattern of the stencil computation.
- Apply HLS optimizations to reduce redundant memory accesses.

  **Questions:**
  - Which data elements are reused across loop iterations?
  - Which arrays become performance bottlenecks?
  - How does array partitioning affect memory bandwidth?
  - What happens if arrays are not partitioned?

### 3. **Loop Structure and Compute Pattern Study**
- Explore different loop organizations or rewrites of the computation.

  **Questions:**
  - Which loop is most effective to pipeline?
  - How does loop reordering affect performance?
  - Can removing or restructuring intermediate copies improve latency?

### 4. **Fixed-Point Data Type Study**
- Understand the impact of using fixed-point arithmetic.

  **Questions:**
  - Why are wider accumulator types needed?
  - How do fixed-point widths affect resource usage?
  - What differences do you observe compared to floating-point designs?

### 5. **Scalability Study**
- Study how performance scales with increasing grid size or number of time steps.

  **Questions:**
  - Does your optimization strategy still work as the problem size grows?
  - Which optimizations scale well, and which do not?

---

## What to Submit

### **1. Your optimized `top.cpp`**
- We will drop this file into the original folder and test correctness and performance.  
- **Your kernel must pass the correctness check** in `host.cpp` to be eligible for ranking and full credit.

### **2. A simple report (in a single `PDF`) containing:**
- Screenshot of HLS synthesized latency
- Screenshot of C/RTL co-simulation latency and clock period
- Screenshot of post-implementation resource utilization
  - Show LUT, FF, BRAM, DSP usage
  - The implementation must complete synthesis, placement, and routing without errors
- **Your final speedup ratio**  
  - Computed relative to the provided baseline
  - Use `number_of_clock_cycles * clock_period` to compute latency

## When to Submit

This lab uses a **two-deadline** system to encourage early correctness and iterative optimization:

- **Initial deadline:**  
  Submit a functionally correct `top.cpp` with a **speedup > 1** (i.e., do not submit the unoptimized baseline).  
  Missing this deadline results in a **30% score penalty**.

- **Final deadline:**  
  Keep optimizing and submit your **best version** by this date.  
  Your best result before the final deadline determines your ranking.

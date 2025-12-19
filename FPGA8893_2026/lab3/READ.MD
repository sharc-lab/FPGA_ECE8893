# Lab 3 – System-Level Optimization with Streaming Dataflow

## Overview

This lab focuses on **system-level optimization** using **streaming dataflow** in High-Level Synthesis (HLS).

Unlike Labs 1 and 2, where performance improvements mainly come from optimizing individual loops or memory access patterns, **Lab 3 requires you to restructure the computation into a streaming pipeline** using:

- `hls::stream`
- `#pragma HLS DATAFLOW`

The baseline implementation provided in `top.cpp` is **correct but intentionally not written as a dataflow design**. Your task is to **refactor and redesign** the computation so that multiple stages can execute **concurrently**.

---

## Key Learning Objectives

By completing this lab, you should understand:

- How to decompose a computation into multiple pipeline stages
- How to use `hls::stream` to connect producer–consumer kernels
- How `#pragma HLS DATAFLOW` enables task-level parallelism
- How FIFO depth and backpressure affect system throughput
- Why optimizing a single kernel is often insufficient for overall performance
- How to balance multiple stages to maximize end-to-end throughput

---

## Computation Structure

The computation consists of **five conceptual stages**:

1. **Preprocess** (per-element)
2. **Transform** (per-element, sliding window)
3. **Statistic Computation** (per-block reduction)
4. **Join + Normalize** (rate mismatch, block-aligned)
5. **Postprocess + Store**

Important characteristics:

- The design includes a **fork**: the same data must feed multiple stages.
- The design includes a **join**: one stage consumes streams with different rates.
- A per-block statistic is produced **only after an entire block is consumed**.
- The join stage must correctly align and apply this statistic to the corresponding block of elements.

This structure is **not a canonical 1-producer / 1-consumer pipeline** and **cannot be efficiently optimized without streaming and buffering**.

---

## Required Design Approach

To achieve good performance, your implementation is expected to:

- Use **`hls::stream`** to connect pipeline stages
- Use **`#pragma HLS DATAFLOW`** at the top level to enable concurrent execution
- Introduce **stream splitting or duplication** where necessary
- Handle **rate mismatch** between per-element and per-block streams
- Use buffering to align data correctly at join points
- Balance pipeline stages so that no single stage dominates throughput

> **Important:** Simply adding `PIPELINE` or `UNROLL` pragmas to the baseline code is not sufficient to achieve high performance in this lab.

---

## Performance Considerations

- The overall throughput is limited by the **slowest stage** in the pipeline.
- Optimizing only one stage may result in **little or no improvement** if other stages remain bottlenecks.
- FIFO depth selection can significantly impact performance and deadlock behavior.
- Successful designs require **both correctness and balanced throughput**.

---

## What You May and May Not Do

### You May:
- Refactor the computation into multiple functions
- Introduce `hls::stream` objects
- Add `DATAFLOW`, `PIPELINE`, `UNROLL`, and other HLS pragmas
- Rewrite the computation order as long as correctness is preserved
- Only change the `top.cpp` file

### You May Not:
- Change the top-level function signature
- Change data types or constants defined in `dcl.h`
- Change the functional behavior of the computation
- Modify the golden reference or host code

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

---

## Final Notes

This lab is intentionally open-ended and challenging.

There is **no single correct architecture**. Strong solutions differ in:
- how data is buffered and split
- how stages are balanced
- how aggressively parallelism is exploited

Your goal is not just to make one kernel fast, but to design a **high-throughput streaming system**.

Good luck, and think like a system architect.

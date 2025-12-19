# Lab 4 – Open-Ended Accelerator Design (Capstone)

## Overview

Lab 4 is the **capstone project** of ECE 8893.

In this lab, you will **design, implement, and optimize your own FPGA accelerator**. Unlike Labs 1–3, **the computation itself is not provided**. You are responsible for:

- defining a meaningful computation,
- implementing a correct baseline,
- designing an optimized accelerator,
- and achieving the highest possible speedup over your own baseline.

This lab is graded **purely by ranking**.

---

## Key Requirements

### 1. Self-Designed Computation

You must design your own computation kernels, subject to the following constraints:

- Your design must consist of **at least five (5) distinct computational kernels**.
- Each kernel must contain **at least 30 lines of code**.
- The overall benchmark must consist of **at least 200 lines of code**.
- The computation must be **meaningful and nontrivial**, involving real data processing (e.g., transforms, reductions, pipelines, or multi-stage workflows), with nontrivial data dependencies (i.e., not independent or dummy stages).


There is **no restriction** on the application domain, as long as the computation is reasonable for FPGA acceleration.

---

### 2. Baseline Definition (**Very Important**)

You must provide a **baseline implementation** against which speedup is measured.

The baseline must satisfy **all** of the following:

- Implement **the same computation** as the optimized design.
- Use the **same kernels**, data types, and functional behavior.
- Use straightforward, sequential code, without HLS performance optimizations.

#### The baseline must **NOT** include:
- Any HLS pragmas.
- Manual buffering or restructuring intended to **slow down** the baseline.
- Artificial delays, empty loops, dummy computation, or intentionally useless code.

> The baseline should represent a reasonable, naïve implementation of the same computation and must not include intentionally unproductive operations, artificial slowdowns, or useless code. Baselines that violate this principle will be treated as cheating and will receive **zero credit for the entire lab**.


---

### 3. Optimized Design

Your optimized implementation should:

- Preserve **exact functional correctness** relative to the baseline.
- Use HLS performance optimizations as appropriate, such as:
  - pipelining,
  - dataflow,
  - streaming,
  - buffering,
  - loop restructuring,
  - parallelism.
- Successfully complete:
  - C synthesis,
  - C/RTL co-simulation,
  - RTL implementation (place and route).

Designs that fail implementation will receive **no credit**, regardless of performance.

---

### 4. Data Types

- Your design must use **fixed-point arithmetic** (`ap_fixed`).
- Data types must be consistent between baseline and optimized implementations.
- Changing data types solely to inflate speedup is **not allowed**.

---

## Performance Metric and Ranking

- Performance is measured using **cycle count** from **C/RTL co-simulation**.
- **Speedup** is defined as:

    `speedup = baseline_cycles / optimized_cycles`


- All submissions will be **ranked globally by speedup**.


You are expected to:
- choose computations with real optimization potential,
- design reasonable baselines,
- and apply aggressive yet correct optimizations.

---

## What You May and May Not Do

### You May:
- Design any computation that satisfies the kernel and baseline requirements.
- Refactor code freely between baseline and optimized versions.
- Use HLS pragmas and streaming in the optimized design.
- Use LLM tools (e.g., ChatGPT, Copilot) to assist with code generation or optimization ideas.

### You May Not:
- Artificially slow down the baseline.
- Change the computation between baseline and optimized designs.
- Include meaningless kernels or placeholder stages.
- Modify correctness checks or benchmarking methodology.

---

## What to Submit

1. Baseline implementation.
2. Optimized implementation.
3. Host code / testbench verifying correctness between the two implementations.
4. Performance report (PDF), including:
 - Screenshot of baseline cycle counts and optimized cycle counts from C/RTL co-simulation.
 - Screenshot of post-implementation resource utilization and clock period.
 - Achieved speedup.

---

## When to Submit (**Very Important**)

This lab uses a **three-stage submission policy**.

### **Deadline 1 – Baseline & Correctness**
- Submit a complete **baseline** and a **functionally correct** optimized design.
- Optimized design does not need to be fast yet, but the speedup ratio must be larger than 1.
- Missing this deadline results in a **20% grade penalty**.

---

### **Deadline 2 – Implementation & Performance**
- Design must pass synthesis, C/RTL co-simulation, and RTL implementation.
- Missing this deadline results in another **20% grade penalty**.

---

### **Deadline 3 – Final Submission**
- Submit your **best optimized design** and final performance results.
- Only the final submission is used for ranking.

---


## Grading Policy

- Lab 4 accounts for **40% of the final course grade**.
- Grading is **purely ranking-based**, determined by achieved speedup.
- Designs that are incorrect, fail implementation, or violate baseline rules will be ranked at the bottom.

---

## Final Notes

Lab 4 is intentionally open-ended and challenging.

There is **no single correct answer**. Both your **design choice** and **optimization techniques** matter.

Good luck — and design something worth optimizing.

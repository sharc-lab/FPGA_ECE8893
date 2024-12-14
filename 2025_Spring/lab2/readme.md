# Lab 2: Efficient Implementation of Attention Computation

## Objective

The attention mechanism is a core component of Transformers and Large Language Models (LLMs) that drives their power and scalability. By computing a weighted combination of input features, attention enables efficient parallelization, context-aware representations, and versatility.

In this lab, we will:

1. Implement the `compute_attention` function for a scaled dot-product attention mechanism in HLS.
2. Load input tensors from binary files and compare your computed output tensor with the provided reference output.
3. Optimize your implementation to minimize latency while staying within resource constraints.

---

## Problem Statement

Your task is to implement the `compute_attention` function, which performs scaled dot-product attention:

$$
\text{Attention}(i, j) = \text{Softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \times V
$$

### Inputs:

- Tensors (`Q`, `K`, `V`) provided as binary files `Q_tensor.bin`, `K_tensor.bin`, and `V_tensor.bin`.


### Outputs:

- Compute the output tensor (`Output_HLS`), compare it against the provided reference tensor (`Output_tensor.bin`), and compute the `MSE`.

### Reference Implementation:

- A reference implementation (unoptimized) is provided as a baseline. Your goal is to optimize this implementation for latency while ensuring correctness and hardware implementability.

---

## Design Constraints

### Must:

1. **No DRAM Partitioning**:
   - Avoid partitioning arrays in DRAM.
2. **Place-and-Route Success**:
   - Your design must be implementable and generate a valid bitstream.
3. **Use Fixed Dimensions**:
   - Do not change `B`, `N`, `dk`, or `dv`.
4. **Report Accurate Latency**:
   - Report latency using either **C/RTL co-simulation** or **LightningSim**. Note: Running co-simulation can take 1~2 hours, so using [**LightningSim**](https://github.com/sharc-lab/LightningSim) is highly encouraged.
   - You may want to start with smaller matrices in early stage for faster exploration.


### Tpis and Notes:

1. **Data Loading**:
   - Try to enforce AXI burst mode and utilize full DRAM bandwidth.
   - Try to use efficient streaming or ping-pong buffering to load data into BRAM.
2. **Parallelization**:
   - Use **dataflow and streaming** to overlap data loading, computation (matrix multiplication, softmax), and writing results.
3. **Softmax Optimization**:
   - Simplify or approximate the softmax computation to reduce latency.
4. **Latency Analysis**:
   - Identify bottlenecks and optimize them (e.g., improve softmax efficiency or scaling operations).
5. **LightningSim**:
   - Use the LightningSim tool from Sharc Lab to quickly estimate total latency and compare results.
6. When printing ap_fixed type values using "%f", remember to add .to_float(); otherwise it always prints zero.

---

## Deliverables

1. **Source Code**:
   - Provide well-documented HLS source code (`.cpp` and `.h` files).
   - Provide HLS synthesis report, C/RTL co-simulation report (or LightningSim screenshot), and post-implementation resource utilization report.
2. **Bitstream File**:
   - Submit the generated `.bit` file to confirm your design is implementable.
3. **Technical Report**:
   - Include the following in your report:
     - Key optimizations applied.
     - Resource utilization (LUTs, BRAMs, DSPs, FFs).
     - Identified bottlenecks and how they were resolved.
     - Post-implementation latency.
     - Comparison between computed and reference outputs.
     - Anything else worth discussing.

---

## Evaluation Criteria

### 1. Implementation Score (80%)

Your implementation will be scored as follows:

- **Correctness** ($a$):
  - $a = 1 - MSE$, where $MSE$ is the Mean Squared Error compared to the reference output.
  - Small error may occur if softmax approximations are applied.

- **Implementability** ($b$):
  - $b = 1$ if the design can place-and-route and generate a valid bitstream.
  - $b = 0$ otherwise.

- **Speedup Ratio** ($s$):
  - $s = \frac{\text{baseline\_latency}}{\text{your\_latency}}$, where your latency is compared against the unoptimized baseline latency.

- **Relative Speedup Score** ($r$):
  - Competes within the class:
    
    $$r =
    \begin{cases} 
    0.2 & \text{if } s < 10 \\
    0.6 + 0.4 \times \left(1 - \frac{\text{your\_rank} - 1}{\text{total\_submission} - 1}\right) & \text{otherwise}
    \end{cases}
    $$

- **Bonus** ($e$)
  - $e = \text{max} (0, 6 - \text{your\_rank})$

- **Final Implementation Score**:
  $f = \max(80, r \cdot b \cdot a) + e$

---

### 2. Report Score (20%)

Your report will be manually graded based on:

- Explanation of optimizations.
- Resource utilization analysis.
- Bottleneck identification and resolution.
- Clear presentation of results and comparisons.
- Use IEEE double-column latex format.

---

## Submission

Submit the following files:

1. **Source Code**: `.cpp` and `.h` files.
2. **Bitstream File**: `.bit` file.
3. **Report**: `.pdf` file.

---

Good luck with your implementation!

# ECE 8893 / Advanced FPGA Programming  
*Georgia Tech — Spring 2026*  
*Instructor: Prof. Cong “Callie” Hao*

### Course Schedule, Sign-up Sheets, and Leaderboards for Labs: [Google Sheets](https://docs.google.com/spreadsheets/d/1spuqURsfj5vi8TcwEBRE93xImNIR5sKu3sXLidmMI_g)

---

## Course Philosophy

ECE 8893 is a **hands-on, performance-driven** course on designing and optimizing FPGA accelerators using High-Level Synthesis (HLS).

There are **no exams**, **no long reports**, only very light quizzes.
Your grade reflects **how fast, correct, and implementable** your hardware designs are.

Think of this class as:

**build hardware → measure → optimize → repeat**  
(with plenty of debugging, confusion, and maybe a few tears along the way).

---

## Who This Course Is For

This course is for you if you:
- like building things that actually run (on real-ish hardware flows),
- don’t mind debugging toolchains (even when they make you want to smash your laptop),
- enjoy performance tuning and iteration (including lots of `why? why! WHY?!?!?`),
- are okay with a little scary (but friendly) competition.

This course may **not** be a great fit if you:
- want step-by-step hand-holding labs,
- want fixed “one correct answer” assignments,
- prefer grades mostly based on writing,
- hate C++, hate debugging, hate open-ended problems, or hate competition.

(Yes, this is your warning 😄)

---

## Grading Policy

This course uses a **performance-based, ranking-style** grading system.

Your score depends on how fast and efficiently your FPGA design runs **relative to the class**, as long as:
- it passes correctness checks, and
- it successfully completes implementation (place & route).

### Grade breakdown
- 93% from Measured Performance (Ranking Style). Each lab is ranked based on achieved speedup. Grades are assigned by **ranking**, not absolute performance thresholds.
- 5% from guest lecture quizzes
- 2% from student presentation attendance

### Expected (approximate) distribution:
- **Top 80% → A**  
- **Next 15% → B**  
- **Bottom 5% → C**

---

## Multi-Deadline System (Per Lab)

Lab 1 - 3 have **two deadlines** to encourage early correctness *and* iterative optimization.

### 1) Initial Submission Deadline (Correctness Checkpoint)

- You must submit a version of `top.cpp` that passes the correctness test in `host.cpp`.
- Your **initial speedup must be > 1**  
  (i.e., you may *not* submit the unmodified baseline).
- If you miss this deadline, a **30% penalty** is applied to your lab score.

### 2) Final Submission Deadline (Optimization Deadline)

- You may continue improving and resubmitting until this deadline.
- Only your **best-performing valid submission** before the final deadline counts for ranking.

Lab 4 has **three deadlines**, one extra to submit the first version of the baseline design (you may change it slightly later).

---

## Implementation Requirement

For the final submission of **any lab** to receive **any score**:

- The design must be functionally correct, passing the provided testbench.
- The design must successfully **synthesize, place, and route** (no implementation errors).
- You must provide a screenshot of **post-implementation resource usage**, including:
  - LUT
  - FF
  - BRAM
  - DSP

If **either correctness or implementation fails**, the score for that lab is **0**, regardless of speedup.

---

## Lab Weights (93% of the final score)

- **Lab 0:** 0% (ungraded, just to get familiar with the flow)
- **Lab 1:** 10%  
- **Lab 2:** 20%  
- **Lab 3:** 30%  
- **Lab 4:** 40%  

---

## Lab Score Calculation

Each lab is scored based on **performance, timeliness, and implementation success**:

### Step 1: Measure Speedup
- Speedup is computed as:
  - **baseline latency ÷ your optimized latency**
- Latency is computed as `number_of_clock_cycles * clock_period`
  - Both cycle counts and clock frequency (period) matter

---

### Step 2: Normalize by the Class
- Your speedup is compared to the **class median speedup** for that lab.
- This normalization helps account for differences in lab difficulty.

---

### Step 3: Initial-Deadline Penalty
- If you submit a **correct** solution by the initial deadline:
  - **No penalty**
- If you miss the initial deadline:
  - **30% penalty** is applied to your lab score

---

### Step 4: Implementation Check
- If your final design **successfully synthesizes and completes place & route**:
  - Your score is valid
- If implementation fails:
  - **Score for that lab is 0**, regardless of performance

---

### Step 5: Final Lab Score

Your final lab score combines:
- normalized performance,
- deadline penalty (if any),
- and implementation validity.

In short:

> **Fast + correct + on-time + implementable designs win.**

---

## Labs Overview

There are one ungraded lab and **four graded labs**:

0. **Lab 0 — Getting to Know the HLS Flow**  
   Learn how to run HLS tools, simulate, synthesize, implement, and read reports.

1. **Lab 1 — Basic Loop Optimizations**  
   Warm-up optimization: rewrite loops, find bottlenecks, get comfortable with basic pragmas.

2. **Lab 2 — Loop & Memory Optimization**  
   Tiling, unrolling, partitioning, buffering, and performance tuning.

3. **Lab 3 — Multi-Kernel / System-Level Accelerator**  
   Streaming, dataflow, fork–join pipelines, and balancing stages so one kernel doesn’t ruin your whole day.

4. **Lab 4 — Open-Ended Final Benchmark (Capstone)**  
   You design your own accelerator, define a baseline, optimize aggressively, and compete for the highest speedup.

All labs must pass correctness tests to be ranked.

---

## Collaboration Policy

All assignments and labs are **individual**.

- Discussing **ideas, strategies, and optimization concepts** is encouraged.
- **Do not share code** or submit others’ implementations.
- Submissions are checked for **code similarity**.

---

## Tools (and AI Tools!)

We will use:
- **Vitis and Vivado**
- Provided build and benchmarking scripts
- Optional advanced tools introduced in class (LightningSim, Allo, TAPA)

We *strongly encourage* the use of AI tools such as:
- ChatGPT / Claude
- GitHub Copilot
- etc.

You may use them to:
- generate or rewrite HLS code,
- propose optimizations,
- debug,
- create benchmarks,
- analyze performance bottlenecks.

**Important:** AI-generated code often breaks, fails synthesis, or does something “creative” (not in a good way).  
You are responsible for **understanding, debugging, and validating** everything you submit.

Learning how to work *with* LLMs (instead of fighting them) is part of this course.

---

## How to Succeed

- Start early — synthesis and implementation time grows fast.
- Get correctness first; optimize second.
- Measure often; trust data, not vibes.
- Try multiple strategies (pipelining, buffering, dataflow, tiling).
- Understand pragmas at an **architectural level** — don’t just sprinkle them and pray.
- Use LLMs intelligently — as assistants, not oracles.

---

## Final Note

ECE8893 this semester is designed to be **project-based, competitive, and creative**.

Your grade depends on **how far you can push the hardware**, not on memorizing definitions.  
Expect iteration, debugging, and performance tuning… and expect to leave with **real FPGA acceleration skills**.

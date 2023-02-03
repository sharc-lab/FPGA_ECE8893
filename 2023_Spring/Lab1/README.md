# Lab 1 Instructions

The goal of this lab is to design accelerators for matrix multiplication. There are two parts to this lab:
- In Part A, you will optimize an existing implementation of matrix multiplication with real values. 
- In Part B, you will implement matrix multiplication with complex values and optimize it. 

In either part, we will multiply a $100\*150$ matrix with a $150\*200$ matrix to get a $100\*200$ matrix product and accelerate the design for Pynq-Z2 board.

**Note**: Part A is very similar to Lab 1 of previous iteration of this course. Please read the policy for *Academic Integrity and Honor Code* below to ensure you don't fall into any troubles.

## Part A: Real Matrix Multiplication (30 points)
Matrix multiplication is at the heart of virtually all deep learning applications and has high scope for parallelism. Compared to a trivial implementation, the amount of parallelism that you can exploit is constrained by the hardware resources and how well your code is optimized. 

In this part of the lab, you are provided with the trivial (simplest) implementation along with a testbench to verify functionality. 
You need to optimize the design to achieve at least **10x** speed-up while ensuring the resource utilization are all under 100%.
- `PartA/real_matmul.cpp` contains the matrix multiplier, which is the top module of the accelerator. This is what you want to modify.
- `PartA/main.cpp` is the testbench. Do not change this file.

**Reference Material**: Ryan Kastner et al., [Parallel Programming for FPGAs](https://github.com/KastnerRG/pp4fpgas/raw/gh-pages/main.pdf), Chapter 7.

## Part B: Complex Matrix Multiplication (70 points)
Many applications in scientific computing and signal processing require working with not just the magnitude but also the phase. This information is aptly captured using complex numbers. Let's build on Part A and develop an accelerator to perform matrix multiplication with complex numbers.

In this part of the lab, you are **not** provided with the trivial implementation. You need to implement the design first (can be trivial), ensure its functional correctness with the testbench provided, and then optimize the design to achieve at least **10x** speed-up while ensuring the resource utilization are all under 100%.
- `PartB/complex_matmul.cpp` is the top module of the accelerator that you need to modify.
- `PartB/main.cpp` is the testbench. Do not change this file.

**Reference Material**: [Complex Matrix Multiplication](https://mathworld.wolfram.com/ComplexMatrix.html)

## How to Run the Codes

In our HLS tutorial, we used GUI for synthesis; However, for faster development, we shall use `Makefile` and `tcl` script for simulation and synthesis.

### Step 1: C Simulation

In `PartA` or `Part B`, go to the relevant folder and just type `make`. Then type `./result` to test if your Vitis HLS function is functionally correct. 

*Important*: Please run C simulation after every change you make to your top function. Cannot stress this enough!

### Step 2: C Synthesis

To run C synthesis, you can either use GUI like we did in the tutorial, or you can use `Makefile` again. Go to the relevant folder (`PartA` or `PartB`) and type `make synth`. This internally runs:
```
vitis_hls script.tcl
```

Once synthesis completes (this shouldn't take more than a minute or two!), you can either open the GUI to read the reports, or you can find the reports under `./real_proj/solution_1/syn/report/` or `./complex_proj/solution_1/syn/report/` folders. The reports with `.rpt` extensions can be opened using text editors.

We recommend using `csynth.rpt` to assess the overall latency and resource utilization (BRAM, DSP, FF and LUT). You can check other reports for analysis purposes.

### Step 3 (optional only if you have an FPGA): Export the design, import in Vivado, build the bitstream, and test on-board!
*How I wish each of us can have a real FPGA board to play with!*

## What to Submit for this Lab

### Part A
1. (20 points) Optimized source code and top-level synthesis report:
    - `real_matmul.cpp`
    - `real_csynth.rpt` (Please rename `./real_proj/solution_1/syn/report/csynth.rpt` to `real_csynth.rpt` before submitting)

2. (10 points) A brief report including:
    - The baseline latency and resource utilization
    - The optimized latency you achieved; how much speed up you gained?
    - The resource utilization
    - What are the main techniques you adopted?

### Part B
1. (50 points) Optimized source code and top-level synthesis report:
    - `complex_matmul.cpp`
    - `complex_csynth.rpt` (Please rename `./complex_proj/solution_1/syn/report/csynth.rpt` to `complex_csynth.rpt` before submitting)

2. (20 points) A brief report including:
    - The baseline latency and resource utilization
    - The optimized latency; how much speed up you gained compared to baseline?
    - The resource utilization (after optimization)
    - What are the main techniques you adopted?

*Note*: Please combine your Part A and Part B reports in a single file and submit `Lab1_Report_<Name>.pdf`. There is no template to follow, however, you are expected to write your report like a research paper.

### Bonus (up to 20 extra points)    
Perform design space exploration (DSE) for PartA or PartB or both.
- Briefly describe the optimization techniques you implemented, the trade-offs in latency and resource that you observed, etc.
- Double the values of `M`, `N`, and `K`. Check how this affects the resource usage and performance. How about the run-time of the tool? Consisely mention your observations.

Submit your analysis and observations in the same report file.

## Academic Integrity and Honor Code
You are required to follow Georgia Tech's [Honor Code](https://policylibrary.gatech.edu/student-life/academic-honor-code) and submit your original work in this lab, especially for Part A. You can discuss this assignment with other classmates but you should code your assignment individually. You are **NOT** allowed to see the code of (or show your code to) other students.

Furthermore, you should **NOT** be looking at the solutions provided in the previous iteration of this course. We will be using the Stanford MOSS tool to detect plagiarism. When there is reasonably clear evidence of a violation, a referral to the Office of the Dean of Students will occur, and all hearings and other resulting procedures will be followed to completion.

## Submission Guideline

Submission: on Canvas for course students, via email (to TA) for Special Problems students

Due date: **Feb. 4 (Sat), 11:59 PM, no extension**.

## Acknowledgements
Thanks to Adou Sangbone Assoa, PhD student with Prof. Arijit Raychowdhury and a former student of this course, for his idea of developing an accelerator for complex matrix multiplication!

Thanks to Ashwin Bhat, also a PhD student with Prof. Arijit Raychowdhury and former student of this course, for his inputs in developing this lab!

## Grading Rubric
$$
Speedup = \frac{Baseline\ Latency}{Optimized\ Latency}
$$

**Note:** Unless simulation test passes, any synthesis results you report have no value. If simulation failure cases are observed, evaluation will be done on a case-to-case basis and partial points may be awarded depending on the degree of simulation error. 

### Part A.1 (20 points)

> simulationTestPass &rarr; +5 points   
> **if** (simulationTestPass):  
> &nbsp;&nbsp;&nbsp;&nbsp; **if**(speedup &geq; 10x), +7 points  
> &nbsp;&nbsp;&nbsp;&nbsp; **else if**(2x &leq; speedup < 10x), +4 points  
> &nbsp;&nbsp;&nbsp;&nbsp; **else**, +2 points  
>   
> &nbsp;&nbsp;&nbsp;&nbsp; **for** resource **in** [BRAM, DSP, FF, LUT]:  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+2 points **if** utilization &leq; 100%  

### Part A.2 (10 points)

> Missing or incomplete information &rarr; -1 point for each question  
> Report values different from the ones achieved in **Part A.1**  &rarr; -2 points each   
> Insufficient description of technique(s) adopted &rarr; -2 points   

### Part B.1 (50 points)

> simulationTestPass &rarr; +15 points   
> **if** (simulationTestPass):  
> &nbsp;&nbsp;&nbsp;&nbsp; **if**(speedup &geq; 10x), +15 points  
> &nbsp;&nbsp;&nbsp;&nbsp; **else if**(2x &leq; speedup < 10x), +10 points  
> &nbsp;&nbsp;&nbsp;&nbsp; **else**, +5 points  
>   
> &nbsp;&nbsp;&nbsp;&nbsp; **for** resource **in** [BRAM, DSP, FF, LUT]:  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+5 points **if** utilization &leq; 100%  

### Part B.2 (20 points)

> **if** baseline latency is not **~2x** or **~4x** of **PartA** baseline &rarr; -5 points unless justified  
> Missing or incomplete information &rarr; -2 point for each question  
> Report values different from the ones achieved in **Part A.1**  &rarr; -3 points each   
> Insufficient description of technique(s) adopted &rarr; -2 points  

### Part C (Extra 20 points)

Awarded on a case-to-case to basis depending on how well the analysis/observations/implementations adhere to the questions stated.

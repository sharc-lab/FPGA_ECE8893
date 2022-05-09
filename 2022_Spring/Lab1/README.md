# Lab 1 instructions

This lab to design an accelerator for a basic matrix multiplication. The simplest implementation and the testbench are provided.
- `top.cpp`: contains matrix_mat function, which is the top module of the accelerator. This is what you want to modify.
- `main.cpp`: testbench. Do not change this file.

## How to run the code

In our previous tutorial, we used GUI for synthesis; now we will try to use makefile and script, which is faster in the future.

### Step 1: C simulation

Just type `make` and `./result` to test if your Vitis HLS function is correct. 

*Important*: please constantly run C simulation after every change you have made to matrix_mat function.

### Step 2: C synthesis

To run C synthesis, you can either use GUI like we did in the tutorial, or you can use the following command line:
```
lastyear vitis_hls -f script.tcl
```

After synthesis, you can either open the GUI to read the reports, or you can find the reports under `./project_1/solution_1/syn/report` folder. The reports ended with `.rpt` can be opened using text editors.

### Step 3 (optional only if you have an FPGA): export the design, import in Vivado, build the bitstream, and test on-board!
*How I wish each of us can have a real FPGA board to play with!*

## What to submit for this lab

1. A brief report including (6 points):
    - The optimized latency you achieved; how much speed up you gained?
    - The resource utilization
    - What are the main techniques you adopted?

2. Source code including (4 points):
    - The modified `top.cpp`
    - The synthesis report ended in `.rpt`
    
3. Bonus for design space exploration (DSE) (up to 2 points):
    - Did you observe or try to apply any trade-offs between latency and resource? 
    - Did you try different quantization methods, and how do they affect the result precision and resource usage?
    - Anything else you observed or want to discuss?

## Submission guideline

Submission: on Canvas

Due date: Feb. 8, 11:59 pm, no extension

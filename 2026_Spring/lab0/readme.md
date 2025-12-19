# Lab 0 – Getting Familiar with the HLS Flow

## Overview

Lab 0 is a warm-up ungraded lab designed to help you get familiar with the Vitis HLS workflow, tools, and reports.
Make sure you can run the tools, understand each stage of the HLS flow, and know where to find key reports.
You will reuse this workflow in every later lab.

---

## What You Will Learn

In this lab, you will learn how to:

- Compile and run the C testbench
- Run Vitis HLS from the command line
- Perform:
  - C simulation
  - C synthesis
  - C/RTL co-simulation
  - RTL implementation (place & route)
- Locate and interpret HLS reports, especially:
  - latency
  - initiation interval (II)
  - resource usage

---

## Files Provided

- dcl.h  
  Shared data types and kernel declarations.

- top.cpp  
  HLS kernel implementation (no need to optimize for this lab).

- host.cpp  
  CPU reference implementation and correctness checker.

- script.tcl  
  Vitis HLS script that runs the full HLS flow automatically.

- Makefile  
  Used to compile and run C simulation.

---

## How to Run

### Step 1: C Compilation and C Simulation

First, compile and run the C version of the program.

```
make
./result.out
```

You should see output indicating that the correctness test has passed.

If this step fails:
- Do NOT proceed to HLS.
- Fix compilation or runtime errors first.

---

### Step 2: Run Vitis HLS

Once C simulation passes, run Vitis HLS using the provided TCL script.

```
vitis-run --tcl script.tcl
```

This script will automatically run the following stages:

1. C Synthesis  
   Translates C/C++ code into RTL.

2. C/RTL Co-Simulation  
   Runs a cycle-accurate simulation of the generated RTL.

3. RTL Implementation  
   Performs logic synthesis, placement, and routing; it can take a long time and is normal.

Make sure you understand this script by reading the commands inside it.

---

## Where to Find Reports

After vitis-run finishes successfully, reports are generated in the following directories.

---

### 1. C Synthesis Report

Location:
`project_1/hls/syn/report/top_kernel_csynth.rpt`

What to look for:
- Estimated latency (cycles)
- Initiation interval (II)
- Resource estimates (LUT, FF, BRAM, DSP)

---

### 2. C/RTL Co-Simulation Report

Location: `project_1/hls/sim/report/top_kernel_cosim.rpt`

What to look for:
- Cycle count from RTL simulation
- Confirmation that co-simulation passed

This latency is more accurate than the C synthesis estimate.

---

### 3. Post-Implementation Report (Place & Route)

Location:
`project_1/hls/impl/report/verilog/top_kernel_export.rpt`

What to look for:
- Final resource usage (LUT, FF, BRAM, DSP)
- Confirmation that implementation completed successfully

This report confirms that your design can actually be mapped to an FPGA.

---

## What to Do If Something Fails

- C simulation fails  
  Fix compilation or logic errors in C/C++.

- C synthesis fails  
  Check for unsupported constructs or type issues.

- C/RTL co-simulation fails  
  Usually indicates functional mismatch or top kernel arguments mismatch.

- Implementation fails  
  Often due to excessive resource usage or timing issues.

For Lab 0, just make sure you can run through the entire flow at least once.
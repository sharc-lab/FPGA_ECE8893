# Lab 3 - On-board Implementation of Vitis HLS IPs

The goal of this lab is to get familiarized with the end-to-end Vitis-Vivado-Pynq tool flow. There are two parts to this lab:
- In Part A, you will take the real matrix multiplication "IP" that you developed in Lab 1 and run it through the Pynq flow. 
- In Part B, you will implement, on-board, the tiling-based convolution IP that you developed in Lab 2.

In both parts, we will be working with Pynq-Z2 FPGA boards. You can read more about Pynq-Z2 [here](http://www.pynq.io/board.html).

As a quick primer, Pynq-Z2 comprises of a Processing System (PS) and a Programmable Logic (PL) among other features. PS runs an embedded Linux OS allowing you to use it as a typical computer (similar to Raspberry Pi). The distinguishing element is the PL or the reconfigurable fabric where you run your Vitis HLS IPs. 

PS-PL paradigm is analogous to CPU-GPU host-device architecture. PS is the "Host" whereas PL is the "Device". You access Device via the Host to send and receive data. 

The end-to-end tool flow involves creating an IP in HLS, exporting it to Vivado, generating a valid bitstream, and then writing a host code application using Python on the Pynq-Z2 board.

**Acknowledgement**: Sincere thanks to Dr. Jeffrey Young for providing access to the Pynq-Z2 boards available at CRNCH lab for this course!

## Part A: Real Matrix Multiplication On-Board (30 points)
We are going back to the time when we first started out using Vitis HLS. Remember those days when Vitis seemed like a no-pain all-gain solution to developing ultra-fast hardware? Good ol’ times.

Anyhoo, in this part of the lab, you are going to take the unoptimised real matrix multiplication “IP” from Lab 1A and actually run it on a Pynq-Z2 FPGA board. You will then run an optimised version of the same IP and analyse the gains in performance on-board.

You are provided with the same trivial implementation of real matrix multiplication (`PartA`) as in Lab 1A with a few minor changes that you don't need to worry about.

Everything you need to know to run the tool flow is provided in the tutorial below. You are required to perform the following two additional tasks:
- Verify the functional correctness of your matrix multiplier in Python using randomly initialized NumPy arrays.
- Use an optimized version of the matrix multiplier (if you've attended the lectures, you'd know where to find it) and re-run the entire flow.

**Reference Material**: [Pynq Tutorial by Akshay Kamath](https://akshaykamath.notion.site/Pynq-Tutorial-300cdf7a7c3947fbb8741509def1eb0f).

## Part B: Tiling-based Convolution On-Board (70 points)
More on this after the Spring Break!

## What to Submit?

### Part A
1. (10 points) Results of running your **unoptimized** matrix multiplier on-board:
    - `real_matmul.bit` (Bitstream)
    - `real_matmul.hwh` (Hardware Handoff file)
    - `host.ipynb` (Your Jupyter Notebook)

2. (10 points) Results of running your **optimized** matrix multiplier on-board:
    - `real_matmul_optimized.bit` (Bitstream)
    - `real_matmul_optimized.hwh` (Hardware Handoff file)
    - `host_optimized.ipynb` (Your Jupyter Notebook)

3. (10 points) A brief report including:
    - A comparison of Vitis HLS latency estimate vs the actual latency observed on-board for unoptimized and optimized IPs.
    - A comparison of speed-up observed w.r.t. CPU computation latency in both cases.
    - Why is the on-board latency different than the Vitis estimate in either case? 
    - (ungraded) Your thoughts about the tutorial shared. Any feedback you'd like to share?

### Part B
Will be updated later.

## Submission Guideline

Submission: on Canvas for course students, via email (to TA) for Special Problems students.

Due date for Part A: **Mar. 19 (Sun), 11:59 PM, no extension**. 

Due date for Part B: **TBD**

## Grading Rubric

### Part A.1
> Jupyter notebook runs without any errors &rarr; +10 points  
> Any problem in bitstream or hardware handoff file &rarr; -2 points each  
> Functionality not verified or hardcoded initialization &rarr; -5 points

### Part A.2
> Jupyter notebook runs without any errors &rarr; +10 points  
> Any problem in bitstream or hardware handoff file &rarr; -2 points each  
> Functionality not verified or hardcoded initialization &rarr; -5 points

### Part A.3
> Missing or incomplete or inconsistent information → -3 points for each question 

## Academic Integrity and Honor Code
You are required to follow Georgia Tech's [Honor Code](https://policylibrary.gatech.edu/student-life/academic-honor-code) and submit your original work in this lab. You can discuss this assignment with other classmates but you should code your assignment individually. You are **NOT** allowed to see the code of (or show your code to) other students.

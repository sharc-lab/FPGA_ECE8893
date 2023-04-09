# Lab 3 - On-board Implementation of Vitis HLS IPs

The goal of this lab is to get familiarized with the end-to-end Vitis-Vivado-Pynq tool flow. There are two parts to this lab:
- In Part A, you will take the real matrix multiplication "IP" that you developed in Lab 1 and run it through the Pynq flow. 
- In Part B, you will implement, on-board, the unoptimized tiling-based convolution IP that you developed in Lab 2.
- In Part C, you will implement the optimized version of the IP in Part B.

In all parts, we will be working with Pynq-Z2 FPGA boards. You can read more about Pynq-Z2 [here](http://www.pynq.io/board.html).

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

## Part B: Unoptimized Tiling-based Convolution On-Board (30 points)
Now that you are familiar with the complete Vitis+Vivado+Pynq flow, let's apply what you learnt in Part A of this lab to implement our favorite convolution layer on-board.

In the `PartB` folder, you will find the reference source code for the first 7x7 layer of ResNet-50 (HD) that we implemented in Lab 2. Your task is to take this unoptimized HLS IP (or feel free to use your own, if its functionally correct) and run it on-board as in Part A. You would need to analyze the on-board latency and functionally verify the results using MSE as the metric (as in the C test bench). Sounds simple?

For the most part, it should be. However, you are **not** provided with a template Jupyter host code this time around. You can refer to the one in Part A to start with, but you will need to figure out how to read the binary files, how to allocate Pynq buffers, how to verify output MSE, etc. yourself!

Note that, unlike Part A which dealt with integers, here we are dealing with fixed-point data. And Python has no native fixed-point data type. To assist you in this regard, helper routines (courtesy Rishov Sarkar!) and their sample usage are shared with you in `fixed_point_converter.py`. 

**Hint**: Use 16-bit unsigned integer (`dtype='u2'`) for buffer allocation to ensure conversion to and from fixed-point happens correctly.

The Vitis HLS latency estimate for the reference code is **22.40 seconds** (you should get the same!). How off is the on-board latency though? Can co-simulation help? You are free to explore!

## Part C: Optimized Tiling-based Convolution On-Board (40 points + up to 20 extra points)
Finally, for the last part of the last lab of this course, you would need to re-run the entire flow using an optimized version of the convolution IP.

The same reference code in `Part B` is shared in `Part C` to allow you to modify it with your favorite optimization pragmas. With the right kind of array partitioning and loop pipelining/unrolling on this code, you should be able to achieve a Vitis HLS latency estimate of **under 500 ms**. This is your HLS latency target for this part of the lab!

You are free to make changes to the code - implement ping-pong buffer or data packing or both. Go wild, try to squeeze every bit of optimization that you can. Lower the latency, the better. If you want extra points, this is what you'd have to do. Note that the students in Lab 2 with the Top 10 latencies (to be announced soon) are **ineligible** to receive extra points in this part of the lab. You are still welcome to re-use your HLS code if you'd like!

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
1. (20 points) Results of running the unoptimized convolution layer on-board:
    - `conv_7x7_unoptimized.bit` (Bitstream)
    - `conv_7x7_unoptimized.hwh` (Hardware Handoff file)
    - `host_PartB.ipynb` (Your Jupyter Notebook)

2. (10 points) A brief report including:
    - A comparison of Vitis HLS latency estimate vs the actual latency observed on-board.
    - Why do you think is the on-board latency different than the Vitis estimate? 
    
### Part C
1. (10 points) ```PartC.tar.gz``` that containing all the source code files.

2. (20 points) Results of running the optimized convolution layer on-board:
    - `conv_7x7_optimized.bit` (Bitstream)
    - `conv_7x7_optimized.hwh` (Hardware Handoff file)
    - `host_PartC.ipynb` (Your Jupyter Notebook)

3. (10 points) A brief report including:
    - A comparison of Vitis HLS latency estimate vs the actual latency observed on-board.
    - How did you optimize the reference code to meet the latency target? What were the techniques/pragmas used?
    - Again, how do you justify the variance in on-board latency?

**Note**: Evaluation of Part B and Part C on-board will be first done using grader host code script (to verify functionality of your bitstream) and then using your submitted host code. 

**Note**: Please combine your Part B and C reports in a single file and submit `Lab3BC_Report_<Name>.pdf`. 

## Submission Guideline

Submission: on Canvas for course students, via email (to TA) for Special Problems students.

Due date for Part A: **Mar. 19 (Sun), 11:59 PM, no extension**. 

Due date for Part B & C: **Apr. 29 (Sat), 11:59 PM**. 

**Note:** Extension will be given for up to 2 additional days only in case there is a known OOD issue in accessing Pynq-Z2 boards. No extension beyond **May 1 (Mon), 11:59 PM** for any reason whatsoever. 

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

### Part B.1
> Functionally-correct bitstream (evaluated using grader script) &rarr; +10 points  
> Jupyter notebook runs without any errors &rarr; +10 points  
> Partial points offered on case-to-case basis

### Part B.2
> Missing or incomplete or inconsistent information → -5 points for each question 

### Part C.1
> **if**(simulation test pass and all resources under 100%)    
> &nbsp;&nbsp;&nbsp;&nbsp; **if**(HLS latency &leq; 300 ms), +30 points (i.e. 20 extra points)  
> &nbsp;&nbsp;&nbsp;&nbsp; **else if** (300 ms < HLS latency < 450 ms), +20 points (i.e. 10 extra points)   
> &nbsp;&nbsp;&nbsp;&nbsp; **else**, +10 points

### Part C.2
> Functionally-correct bitstream (evaluated using grader script) &rarr; +10 points    
> Jupyter notebook runs without any errors &rarr; +10 points  
> Partial points offered on case-to-case basis  

### Part C.3
> Missing or incomplete or inconsistent information → -3 points for each question 

## Academic Integrity and Honor Code
You are required to follow Georgia Tech's [Honor Code](https://policylibrary.gatech.edu/student-life/academic-honor-code) and submit your original work in this lab. You can discuss this assignment with other classmates but you should code your assignment individually. You are **NOT** allowed to see the code of (or show your code to) other students.

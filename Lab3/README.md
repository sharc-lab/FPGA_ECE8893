# Lab3: Execute your bitstream on a real FPGA

The goal of this lab is to guide you and allow you to run your designs on a real FPGA. You will have remote access to FPGA boards to implement your designs (thanks to Dr. Jeffrey Young's help!). Check the `How To` section below for details.

## Introduction

The typical end-to-end design flow (from HLS to on-board) involves the following steps:
1. Develop your accelerator IP in HLS C/C++, verify its functional correctness through simulation, synthesize it with Vitis HLS, and analyze its performance using the generated synthesis reports. Repeat these steps as you optimize your design until you achieve a satisfactory performance. (You are already familiar with this step from Lab2!)
2. Run C/RTL co-simulation on your finalized (optimal) design to get a more accurate performance evaluation. **This step is strongly recommended.** If the co-simulation result shows a large gap compared to the HLS reports, it means something is wrong and you would need to go back to Step 1 to resolve (Tip: Look at `vitis_hls.log` for debugging).
3. Export your IP from Vitis HLS, import it in Vivado, connect the IO ports and buses appropriately (to Zynq7 PS), create the HDL design wrapper, and generate the bitstream (Details about exporting IP and generating bitstream can be found in the `tutorial` folder).
4. Upload the bitstream (.bit) and the corresponding hardware description file (.hwh) to Jupyter notebook (rename both files to have a common name!), write your host code in Python, and start the execution of the IP you designed!

## Lab Credits: Total 10 points
This lab is divided into Part A and Part B.

## Part A: Familiarizing with the End-to-end FPGA Flow (5 points)
**Due date: Apr. 3rd, 11:59PM ET**

### Requirement:
- To make your life easier, in part A, you will only implement a simple vector addition IP. This HLS IP should read in a 1-D array of size 100 from DRAM, add 1 to each array element, and write the array back to DRAM.
- Please select `Pynq-Z2` board in your Vitis HLS and Vivado projects. You can also search it with the product code: `xc7z020clg400-1`
- Please name your IP, i.e., the top function name, as `vec_add_<your_last_name>`.
- In your Python host code, you need to *randomly initialize the input array values*. 

### What to submit:
   -  Your host code in Python (.py or .ipynb)
   -  Your generated bitstream (.bit) and hardware description file (.hwh)
   -  A screenshot of your Jupyter notebook output: the first 10 values in the array before and after calling your HLS IP.


## Part B: Implementing your Final Project on FPGA (5 points) 
**Due date: Same as final project**

### Requirement:
- To make your life slightly harder (or still easier?), in part B, you will be implementing your final project design on the FPGA board. As you may not have the time to implement your entire design on-board, it is sufficient if you demonstrate a functionally-correct component of your design working on-board. For example, if you are implementing a DNN, you can choose one (or more) layers for demonstrating and verifying on-board.
- For those who have opted for slightly different final projects that doesn't require actual HLS programming, you need to demonstrate the execution of your Lab2 code on the FPGA board.

### What to submit:
 -  Your host code in Python (.py or .ipynb)
 -  Your generated bitstream (.bit) and hardware description file (.hwh)
 -  A brief report with a screenshot proving that the on-board functionality is correct.
 -  In the report, please put the on-board execution time and compare it with HLS synthesis report. For example, your HLS synthesis report may say 50ms, while your measured on-board latency may be 88ms.

### Something to Keep in Mind...
In Part A, you are working with the standard integer data type. However, in Part B, based on the nature of your final project, you may be working with the **fixed-point** data type (as in Lab2). Now, Python does not have a native fixed-point data type, so a direct implementation like Part A will not suffice for verifying the functionality of your IP. You (Hint: Fixed-point variables are internally integers!)

## How To Connect to Pynq Cluster and Use Jupyter Notebook

1. ```ssh <GT-Username>@synestia2.cc.gatech.edu```
2. Once you are logged in, run ```/net/cs3220_share/student_scripts/init_student_vivado_env.sh```
3. Next, run ```/net/cs3220_share/student_scripts/run-jupyter-pynq.sh```
4. Wait for a job (occasionally prompt by typing a letter?)
5. Follow instructions to forward port and open Jupyter instance on your local browser

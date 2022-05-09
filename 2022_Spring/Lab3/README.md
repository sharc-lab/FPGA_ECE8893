# Lab3: Execute your bitstream on a real FPGA

This lab helps you run your design on a real FPGA. Briefly, here are the steps to follow:
1. Design your accelerator in Vitis HLS, optimize it, and get a synthesis report (this step is iterative until a satisfying performance is achieved)
2. Run C/RTL co-simulation to get a more accurate performance evaluation. This step is strongly recommended. If the co-sim result is showing a large gap from the synthesis report, something is wrong and please go back to step 1.
3. Export IP (which is designed by you) from Vitis HLS, import IP in Vivado, connect ports and buses, generate design wrapper, and generate the bistream (more details about exporting IP and generating bitstream can be found in the `tutorial` folder).
4. Upload the bitstream (.bit) and hardware description file (.hwh) to Jupyter notebook, write your host code in Python, and start the execution of the IP you designed!


## Credits: total 10 points
## Part A: 5 points - due date Apr. 3rd, 11:59pm

### Requirement:
- To make life easier, in part A, let's only implement a simple vector addition. The HLS IP reads in a 1-D array of size 100 from DRAM, add 1 to each element, and write the array back to DRAM.
- Please select `Pynq-Z2` board in your Vitis HLS and Vivado projects.
- Please name your IP, i.e., the top function name, as `vec_add_<your_last_name>`.
- In your host code, please *randomly initialize the input array values*. 




### What to submit:
   -  Your host code in Python (.py or .ipynb)
   -  Your generated bitstream (.bit) and hardware description file (.hwh)
   -  A screenshot of your Jupyter notebook output: the first 10 values in the array before and after calling your HLS IP.


## Part B: 5 points - due date: same as final project

### Requirment:
- To make life harder (or still easier?), in part B, you'll implement whatever you have for your final project on board. It can be a small portion of your whole project, since you probably won't have time to finish everything. For example, if you're implementing the entire DNN, you can choose one layer to put it on board and verify.
- For those whose final project is slightly different and doesn't require actual HLS programming, please put your Lab2 code on the board.

### What to submit:
 -  Your host code in Python (.py or .ipynb)
 -  Your generated bitstream (.bit) and hardware description file (.hwh)
 -  A brief report with a screenshot proving that the on-board functionality is correct.
 -  In the report, please put the on-board execution time and compare it with HLS synthesis report. For example, your HLS synthesis report may say 50ms, while your measured on-board latency may be 88ms.



## How to connect to Pynq cluster and use Jupyter notebook (thanks for Dr. Jeffrey Young's help)

1. ```ssh <GT-Username>@synestia2.cc.gatech.edu```
2. Once log in, run ```/net/cs3220_share/student_scripts/init_student_vivado_env.sh```
3. run ```/net/cs3220_share/student_scripts/run-jupyter-pynq.sh```
4. Wait for a job (occasionally prompt by typing a letter?)
5. Follow instructions to forward port and open Jupyter instance on your local browser


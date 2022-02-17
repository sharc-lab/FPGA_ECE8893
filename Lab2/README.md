<h1>Parallel Programming for FPGAs</h1>
<h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ECE 8893 - FPG</h3>
<h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Spring 2022</h3>
<h2>Lab2: Tiling-based Convolution in HLS</h2>

<h3>Problem Statement:</h3>

Implement the following convolutional layer (sample taken from the ResNet50 Model, refer [1]):
```
Kernel Size 		- 	3x3
Filter Size	 	- 	64
Input Feature Map 	- 	(64, 184, 320) 
Stride 			- 	1
Padding 		- 	1
Output Feature Map      - 	(64, 184, 320)
```
For reference on Convolutional Layer in deep learning, refer [2] 
 
<h4>Part A - C model implementation (30% Weightage):</h4>

Implement the convolution at the software-level (no hardware level features). This is to ensure functional correctness. 

Your code must pass the testbench.

Note: Here, handling the border pixels correctly for padding is very important ( Refer to [2] ) as you would extend this concept for the hardware implementation in the next part.
 
<h4>Part B - Unoptimized but synthesizable tiled-convolution in HLS (30% Weightage):</h4>

Implement the convolution layer as a synthesizable tiled convolution (Refer [1]).<br>
a) Implement the convolution for a single tile in conv_3x3.cpp<br>
b) Implement the complete tiled convolution in tiled_conv.cpp<br>

There are functions available utils.cpp to assist with this.
 

<h4>Part C - Optimized and (synthesizable) tiled-convolution (40% Weightage):</h4>

Apply suitable HLS optimizations to optimize the latency while keeping resource utilization under 100%. The target is to achieve an overall speedup of at least 24x.

You need to estimate how much of a speedup your optimized convolution code should hit to meet this overall speedup. Amdahl’s law in action!

Note: This is the most important part of this lab.
 
This lab can take considerable time and experimentation compared to the previous one so be sure to take a look at it early ahead and schedule time for it accordingly.
 
<h4>References:</h4>

[1] Lab 2 Reference Article:<br>
https://sharc-knowledgebase.netlify.app/articles/cnn/tiling-based_convolution_for_hls/<br>
[2] Conv Layer Reference:<br>
https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
 
<h4>Some constraints to follow:</h4>

For Part B & Part C, buffer dimensions should not be changed except for minor increments in input buffer height and width for handling the border features.<br>
Tile or Tile block dimensions should not be changed.<br>
Input or output feature map dimensions should not be changed.<br>
Clock period should be fixed at 10ns.<br>
The data precision should not be changed for latency calculation (can be explored).<br>
 
<h4>Targets:</h4>

Part A: Functional correctness. Test bench must pass.<br>
Part B: Latency of ~2.24 billion cycles (i.e. ~22.4 seconds) for the entire layer.<br>
Part C: Latency of <93.5 million cycles (i.e <935 ms) for the entire layer.<br>
Resource utilization to be under 100% in both Part B and Part C.<br>
 
<h4>What to submit:</h4>

Part A - `src/model_conv.cpp`.<br>
Part B - `src/conv_3x3.cpp`.<br>
Part B - `src/tiled_conv.cpp`.<br>
Part C. a) ‘‘src/utils.cpp’<br>
Part C. b) optimized `src/conv_3x3.cpp`. HLS report.<br>
Part C. c) optimized `src/tiled_conv.cpp`. HLS report.<br>

<h4>Lab Report:</h4>

Part C. a) single tile convolution latency obtained, resource utilization, techniques applied.<br>
Part C. b) entire layer latency obtained, resource utilization, techniques applied.<br>
Explorations / Observations for single tile convolution optimization.<br>
Explorations / Observations for entire layer convolution optimization.<br>
All code should compile and pass testbench.<br>
 
<h4>Reference Data:</h4>

Input to the convolution:<br>
`bin/conv_layer_input_feature_map.bin`<br>
Golden output of the convolution (for comparing results with):<br>
`bin/conv_layer_output_feature_map.bin`<br>
Weights for the convolution layer:<br>
`bin/conv_layer_weights.bin`<br>
Biases for the convolution layer:<br>
`bin/conv_layer_bias.bin`

These .bin files are generated from Pytorch model. This is the reason why MSE is being used in the testbench for checking correctness (There is a variation that arises due to the difference in the way floating points are handled in C and Python). So the MSE should not be exactly 0.
 


<h4>Grading (Refer 'What to submit'):</h4>

Part A - C model implementation -> 3pts<br>
Part B - Unoptimized by synthesizable tiled-convolution -> 3pts<br>
Part C - Optimized and (synthesizable) tile-convolution -> 4pts<br>
If Lab Report not submitted -> -6pts<br>
If Lab Report submitted but not up to the mark or missing required items -> accordingly pts can be subtracted.<br>
 
<h4>Files:</h4>

tiled_conv -> implement complete tiled convolution<br>
conv_3x3 -> implement convolution computation<br>
model_conv -> implement convolution layer at software level (no hardware level features involved)<br>
sim -> testbench<br>
conv.h -> contains defines and function declarations<br>
utils.cpp -> contains useful functions for implementing tiled convolution<br>

<h4>Scripts (copy to src folder or create symbolic links):</h4>

`scripts/0_cmodel_sim_float.sh` -> run C simulation of the software-level implementation with floating point data type (faster)<br>
`scripts/1_cmodel_sim_fixp.sh` -> run C simulation of the software-level implementation with fixed point data type (slower)<br>
`scripts/2_hls_sim_float.sh` -> run C simulation of the hardware level implementation with floating point data type<br>
`scripts/3_hls_sim_fixp.sh` -> run C simulation of the hardware level implementation with fixed point a type<br>
Run_hls_synth.tcl -> to run Vitis HLS synthesis<br>

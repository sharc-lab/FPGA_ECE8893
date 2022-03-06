<h1>Parallel Programming for FPGAs</h1>
<h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ECE 8893 - FPG</h3>
<h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Spring 2022</h3>
<h2>&nbsp;&nbsp;Lab2: Tiling-based Convolution in HLS</h2>

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

 ---
 
<h4>Part A - C model implementation (30% Weightage):</h4>

&nbsp;&nbsp;&nbsp;&nbsp;Implement the convolution at the software-level (no hardware level features). This is to ensure functional correctness and to act as a debugging reference for implementing tiling-based convolution.

&nbsp;&nbsp;&nbsp;&nbsp;Your code must pass the testbench.

```
Note: Here, handling the border pixels correctly for padding is very important ( Refer to [2] ) 
as you would extend this concept for the hardware implementation in the next part.
 ```
---

<h4>Part B - Unoptimized but synthesizable tiled-convolution in HLS (30% Weightage):</h4>

&nbsp;&nbsp;&nbsp;&nbsp;Implement the convolution layer as a synthesizable tiled convolution (Refer [1]).<br>
&nbsp;&nbsp;&nbsp;&nbsp;a) Implement the convolution for a single tile in conv_3x3.cpp<br>
&nbsp;&nbsp;&nbsp;&nbsp;b) Implement the complete tiled convolution in tiled_conv.cpp<br>
&nbsp;&nbsp;&nbsp;&nbsp;There are functions available in utils.cpp to assist with this.

---

<h4>Part C - Optimized and (synthesizable) tiled-convolution (40% Weightage):</h4>

&nbsp;&nbsp;&nbsp;&nbsp;Apply suitable HLS optimizations to optimize the latency while keeping resource utilization under 100%. The target is to achieve an overall `latency of < 740ms` (speedup of at least `30x`)

&nbsp;&nbsp;&nbsp;&nbsp;You need to estimate how much of a speedup your optimized convolution code should hit to meet this overall speedup. Amdahl’s law in action!

```
Note: This is the most important part of this lab. 
 ```
---


<h4>Bonus!</h4>
Up for the challenge? <br>
- If the overall latency achieved is less than 600ms, 1 extra point. <br>
- If the overall latency achieved is less than 450ms, total 2 extra points. <br>
- To add a little spice to this, if your latency is among the Top 5 in the class, you get 1 additional point. <br>
- If you're in the Top 10 but not in the Top 5, you get 0.5 additional point. <br>
(i.e. you can score a maximum of 13 points in this lab!!) <br>

---

&nbsp;&nbsp;&nbsp;&nbsp;Sugested sequence of code development:<br>
&nbsp;&nbsp;&nbsp;&nbsp;* The software-level C model<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Tiled implementation: tiled_conv + conv_3x3 together<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Optimize conv_3x3<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Optimize tiled_conv<br>
&nbsp;&nbsp;&nbsp;&nbsp;Before running synthesis, always simulate the whole layer first, make sure it passes, then synthesize<br>

```
Note: This lab can take considerable time and experimentation compared to the previous one 
so be sure to take a look at it early ahead and schedule time for it accordingly.
```


---

<h4>References:</h4>

&nbsp;&nbsp;&nbsp;&nbsp;[1] Lab 2 Reference Article:<br>
&nbsp;&nbsp;&nbsp;&nbsp;https://sharc-knowledgebase.netlify.app/articles/cnn/tiling-based_convolution_for_hls/<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2] Conv Layer Reference:<br>
&nbsp;&nbsp;&nbsp;&nbsp;https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
 
 
<h4>Some constraints to follow:</h4>

&nbsp;&nbsp;&nbsp;&nbsp;* For Part B & Part C, buffer dimensions should not be changed except for minor increments in input buffer height and width for handling the border pixels.<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Tile slice or Tile cube dimensions should not be changed.<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Input or output feature map dimensions should not be changed.<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Clock period should be fixed at 10ns.<br>
&nbsp;&nbsp;&nbsp;&nbsp;* The data precision should not be changed for latency calculation (can be explored).<br>
 
<h4>Targets:</h4>

&nbsp;&nbsp;&nbsp;&nbsp;* Part A: Functional correctness. Test bench must pass.<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Part B: Latency of `~22.2 seconds` for the entire layer (This is just a reference for the kind of latency expected and not a target as such).<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Part C: Latency `< 740 ms` for the entire layer(i.e. 30x speedup).<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Resource utilization to be under 100% in both Part B and Part C.<br>
 
<h4>What to submit:</h4>

&nbsp;&nbsp;&nbsp;&nbsp;* Part A - `src/model_conv.cpp`.<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Part B. a) `src/utils.cpp`.<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Part B. b) `src/conv_3x3.cpp`.<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Part B. c) `src/tiled_conv.cpp`.<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Part C. a) `src/utils.cpp`.<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Part C. b) optimized `src/conv_3x3.cpp`. HLS report.<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Part C. c) optimized `src/tiled_conv.cpp`. HLS report.<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Part C. d) `src/conv.h`. <br>
&nbsp;&nbsp;&nbsp;&nbsp;* Lab Report <br>
```
For Part A, upload your model_conv.cpp
For Part B, upload Part_B.tar.gz which contains PartB. a), b) and c) without optimizations.
For Part C, upload Part_C.tar.gz which contains PartC. a), b), c) and d) with optimizations & pragmas
```
<h4>Reference Data:</h4>

&nbsp;&nbsp;&nbsp;&nbsp; * Input to the convolution:<br>
&nbsp;&nbsp;&nbsp;&nbsp; `bin/conv_layer_input_feature_map.bin`<br>
&nbsp;&nbsp;&nbsp;&nbsp; * Golden output of the convolution (for comparing results with):<br>
&nbsp;&nbsp;&nbsp;&nbsp; `bin/conv_layer_output_feature_map.bin`<br>
&nbsp;&nbsp;&nbsp;&nbsp; * Weights for the convolution layer:<br>
&nbsp;&nbsp;&nbsp;&nbsp; `bin/conv_layer_weights.bin`<br>
&nbsp;&nbsp;&nbsp;&nbsp; * Biases for the convolution layer:<br>
&nbsp;&nbsp;&nbsp;&nbsp; `bin/conv_layer_bias.bin`

&nbsp;&nbsp;&nbsp;&nbsp;These .bin files are generated from Pytorch model. This is the reason why MSE is being used in the testbench for checking correctness (There is a variation that arises due to the difference in the way floating points are handled in C and Python). So the MSE should not be exactly 0.
 


<h4>Grading (Refer 'What to submit'):</h4>

&nbsp;&nbsp;&nbsp;&nbsp;* Part A - C model implementation -> 3pts<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Part B - Unoptimized by synthesizable tiled-convolution -> 3pts<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Part C - Optimized and (synthesizable) tiled-convolution -> 4pts<br>
&nbsp;&nbsp;&nbsp;&nbsp;* If Lab Report not submitted -> -6pts<br>
&nbsp;&nbsp;&nbsp;&nbsp;* If Lab Report submitted but not up to the mark or missing required items -> accordingly pts can be subtracted (Refer 'What to include in the Lab Report' below).<br>
&nbsp;&nbsp;&nbsp;&nbsp;* Bonus points can be obtained as described in the 'Bonus' section.


<h4>What to include in the Lab Report:</h4>

&nbsp;&nbsp;&nbsp;&nbsp;Part A (-1pts):<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1) How did you handle the border condition related to padding? <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2) What is the MSE obtained (floating-point simulation)? 
    
&nbsp;&nbsp;&nbsp;&nbsp;Part B (-1pts):<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1) What is the MSE obtained (floating-point simulation)? <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2) What is the latency and utilization (of each resource) you obtained after synthesis (for both single tile case and for the entire layer)? 
```
Note: For these above questions, a brief and precise answer in one or two sentences is enough. For utilization mention resource count.
```
&nbsp;&nbsp;&nbsp;&nbsp;Part C (-4pts):<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1) A table indicating each optimization tried and corresponding latency, speedup, and utilization (of each resource) obtained. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2) A brief description of the experimentation and observations from the exploration. 
```
Note: For Part C. 2) , you may use 10 - 20 sentences to effectively convey your experimentation, observations, and conclusions.
```

<h4>Files:</h4>

&nbsp;&nbsp;&nbsp;&nbsp;* tiled_conv.cpp -> implement complete tiled convolution<br>
&nbsp;&nbsp;&nbsp;&nbsp;* conv_3x3.cpp -> implement convolution computation<br>
&nbsp;&nbsp;&nbsp;&nbsp;* model_conv.cpp -> implement convolution layer at software level (no hardware level features involved)<br>
&nbsp;&nbsp;&nbsp;&nbsp;* sim.cpp -> testbench<br>
&nbsp;&nbsp;&nbsp;&nbsp;* conv.h -> contains defines and function declarations<br>
&nbsp;&nbsp;&nbsp;&nbsp;* utils.cpp -> contains useful functions for implementing tiled convolution<br>
&nbsp;&nbsp;&nbsp;&nbsp;* csim.out -> binary obtained after compilation that can be run to simulate the design


<h4>Scripts:</h4>

&nbsp;&nbsp;&nbsp;&nbsp;* `0_cmodel_sim_float.sh` followed by `./csim.out`-> Compile and simulate the software-level implementation with floating point data type (faster)<br>
&nbsp;&nbsp;&nbsp;&nbsp;* `1_cmodel_sim_fixp.sh` followed by `./csim.out`-> Compile and simulate the software-level implementation with fixed point data type (slower)<br>
&nbsp;&nbsp;&nbsp;&nbsp;* `2_hls_sim_float.sh` followed by `./csim.out`-> Compile and simulate hardware level implementation with floating point data type<br>
&nbsp;&nbsp;&nbsp;&nbsp;* `3_hls_sim_fixp.sh` followed by `./csim.out` -> Compile and simulate hardware level implementation with fixed point a type<br>
&nbsp;&nbsp;&nbsp;&nbsp;* `conv_synth.tcl` -> Run Vitis HLS synthesis on conv_3x3<br>
&nbsp;&nbsp;&nbsp;&nbsp;* `layer_synth.tcl` -> Run Vitis HLS synthesis on tiled_conv<br>

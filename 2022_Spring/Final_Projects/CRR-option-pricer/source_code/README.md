The python code I have is just a simple CRR model to compare my outputs
with my C++ model to make sure my program is correct. You can also
change the parameters to test out other outputs

https://xilinx.github.io/Vitis\_Libraries/quantitative\_finance/2020.2/guide\_L2/engines/BTCRR.html
ref:https://github.com/tienusss/Option\_Calculations
ref:https://sites.google.com/view/vinegarhill-financelabs/home?authuser=0

run: crr.py

//////////////////////////////////////////////////////////////////////
HLS folder: This has my HLS program for running my CRR model. Can do
both C-sim and HLS synthesis setup env using the below commands
/////////////////////////////////// source
/tools/software/xilinx/setup\_env.sh source
/tools/software/xilinx/Vitis\_HLS/2021.1/settings64.sh alias
vitis\_hls="/tools/software/xilinx/Vitis\_HLS/2021.1/bin/vitis\_hls"
alias vivado=\"/tools/software/xilinx/Vivado/2021.1/bin/vivado
///////////////////////////////////////////////////////////////////////
Compile: make

execute C-simulation: ./csim.out

HLS synthesis: lastyear vitis\_hls -f script.tcl

//////////////////////////////////////////////////////


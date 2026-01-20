open_component project_1

# set top function of the HLS design
set_top top_kernel

# add source file
add_files top.cpp

# add testbench
add_files -tb host.cpp

# stop automatic unrolling and pipelining by Vitis so baseline design fits on FPGA
config_unroll -tripcount_threshold 0
config_compile -pipeline_loops 0

# FPGA part and clock configuration
# default frequency is 100 MHz
set_part {xczu3eg-sbva484-1-e}
#create_clock -period 4 -name default

# C synthesis for HLS design, generating RTL
csynth_design

# C/RTL co-simulation; can be commented if not needed
cosim_design

# export generated RTL as an IP; can be commented if not needed
# Note: -flow syn performs RTL synthesis; 
# -flow impl performs both RTL synthesis and implementation, including a detailed place and route of the RTL netlist.
# implementation flow will take much longer time
#export_design -format ip_catalog
export_design -format ip_catalog -flow impl

exit

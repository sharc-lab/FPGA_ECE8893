AUTOPILOT_ROOT := /tools/software/xilinx/Vitis_HLS/2022.1
ASSEMBLE_SRC_ROOT := .

IFLAG += -I "${AUTOPILOT_ROOT}/include"
IFLAG += -I "${ASSEMBLE_SRC_ROOT}"
IFLAG += -I "/usr/include/x86_64-linux-gnu"
IFLAG += -g 

CFLAG += -fPIC -O3 -lm -std=c++11
CC      = g++ 

all:
	g++ *.cpp -o csim.out $(CFLAG) $(IFLAG)

synth:
	vitis_hls script.tcl

clean:
	rm -f *.o result *.log

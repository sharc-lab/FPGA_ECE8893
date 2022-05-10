AUTOPILOT_ROOT :=/tools/software/xilinx/Vitis_HLS/2021.1/

ASSEMBLE_SRC_ROOT := .
TB_ROOT := .
IFLAG += -I "${AUTOPILOT_ROOT}/include"
IFLAG += -I "${ASSEMBLE_SRC_ROOT}"
IFLAG += -I "${ASSEMBLE_SRC_ROOT}" 
IFLAG += -I "/usr/include/x86_64-linux-gnu"
IFLAG += -D__SIM_FPO__ -D__SIM_OPENCV__ -D__SIM_FFT__ -D__SIM_FIR__ -D__SIM_DDS__ -D__DSP48E1__

IFLAG +=  -g -DHLS_SIM
CFLAG += -fPIC -O3
CC      = g++ 


ALLOUT+= csim.out

all: $(ALLOUT) 
##TO BE MODIFIED START

#main.o:./main.cpp
#	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
top.o:./top.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
##TO BE MODIFIED END

#IP_DEP+=main.o
IP_DEP+=top.o

main.o:./main.cpp
	$(CC) $(GCOV)  $(CFLAG)  -I "${ASSEMBLE_SRC_ROOT}" -o $@  -c $^   -MMD $(IFLAG)

csim.out: main.o $(IP_DEP)
	$(CC)  $(GCOV)  $(CFLAG) -MMD $(IFLAG)  -o $@  $^ 

clean:
	rm -f -r csim.d 
	rm -f *.out *.gcno *.gcda *.txt *.o *.d


/* This is the host code for the HLS kernel, used to control, provide inputs, and collect results from HLS kernel */
/* Also known as the testbench, used to check the functionality of HLS kernel */

#include "dcl.h"


int main()
{
	data_8 a[DIM][DIM];
	data_8 b[DIM][DIM];
	data_8 c_HLS[DIM][DIM];
	data_8 c_ref[DIM][DIM];
	
	// Multiply matrix_a_E4M3 with matrix_b_E4M3 and compare with matrix_c_E4M3
	
    std::ifstream inf1("matrix_a_E4M3.bin", std::ios::binary);
	std::ifstream inf2("matrix_b_E4M3.bin", std::ios::binary);
	std::ifstream inf3("matrix_c_E4M3.bin", std::ios::binary);
	

    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            inf1.read(reinterpret_cast<char*>(&a[i][j]), sizeof(data_8));
			inf2.read(reinterpret_cast<char*>(&b[i][j]), sizeof(data_8));
			inf3.read(reinterpret_cast<char*>(&c_ref[i][j]), sizeof(data_8));
        }
    }
	inf1.close();
	inf2.close();
	inf3.close();

	// call HLS kernel
	for(int i = 0; i < DIM; i++) {
		for(int j = 0; j < DIM; j++) {
			c_HLS[i][j] = 0;
		}
	}
	
	printf("Calling HLS kernel to compute matrix C in E4M3...\n");
	MatMul_E4M3(a, b, c_HLS);

	printf("HLS kernel done.\n");

	// compare c_HLS with c_ref

	// do the same for all other matrices
	// ...
	// ...
	// ...



	

	return 0;
}

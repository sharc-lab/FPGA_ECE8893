#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>

int main() {
    // Dimensions
    const int C1 = 256;
    const int C2 = 256;
    const int C3 = 40;
    const int C4 = 40;

    float**** input4D = (float****)malloc(C1 * sizeof(float***));
   
    for (int i = 0; i < C1; ++i) {
        // Allocate memory for C2 pointers to C3
        input4D[i] = (float***)malloc(C2 * sizeof(float**));

        for (int j = 0; j < C2; ++j) {
            // Allocate memory for C3 pointers to C4
            input4D[i][j] = (float**)malloc(C3 * sizeof(float*));

            for (int k = 0; k < C3; ++k) {
                // Allocate memory for C4 floats
                input4D[i][j][k] = (float*)malloc(C4 * sizeof(float));
            }
        }
    }
 


    // Open the binary file
    std::ifstream ifs_param("input_1.bin", std::ios::in | std::ios::binary);
    if (!ifs_param.is_open()) {
        std::cerr << "Error: Failed to open file." << std::endl;
        return -1;
    }

    // Read all the data into the weights array
    ifs_param.read((char*)(***input4D), C1 * C2 * C3 * C4 * sizeof(float));

    // Check if read was successful
    if (!ifs_param) {
        std::cerr << "Error: Failed to read the expected amount of data." << std::endl;
        return -1;
    }

    // Close the file
    ifs_param.close();

    // Print the first 10 values
    std::cout << "First 10 values from input_1.bin:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "input[" << i << "] = " << input4D[0][0][0][i] << std::endl;
    }

    return 0;
}

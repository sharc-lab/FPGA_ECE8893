///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    sim.cpp
// Description: Test bench for C model simulation of 7x7 convolution 
//
// Note:        You are not required to turn in this code. 
//              In case you make any modifications (for debugging, etc.),
//              do ensure your design works with the original test bench.
//
//              For debugging, you may use the PRINT_DEBUG construct.
//              You can put your print statements inside this ifdef block and
//              use "make debug".
///////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <cmath>

#include "conv.h"

using namespace std;

//--------------------------------------------------------------------------
// Convolution layer inputs, parameters, and reference output
//--------------------------------------------------------------------------
float conv_layer_input_feature_map[3][736][1280];
float conv_layer_weights[64][3][7][7];
float conv_layer_bias[64];

float conv_layer_golden_output_feature_map[64][368][640];
float conv_layer_output_feature_map[64][368][640];

//--------------------------------------------------------------------------
// Read the reference files into test bench arrays
//--------------------------------------------------------------------------
void read_bin_files()
{
    // Input Feature Map
    ifstream ifs_conv_input("../bin/conv_input.bin", ios::in | ios::binary);
    ifs_conv_input.read((char*)(**conv_layer_input_feature_map), 3*736*1280*sizeof(float));
    ifs_conv_input.close();

    // Weights
    ifstream ifs_conv_weights("../bin/conv_weights.bin", ios::in | ios::binary);
    ifs_conv_weights.read((char*)(***conv_layer_weights), 64*3*7*7*sizeof(float));
    ifs_conv_weights.close();
    
    // Bias
    ifstream ifs_conv_bias("../bin/conv_bias.bin", ios::in | ios::binary);
    ifs_conv_bias.read((char*)(conv_layer_bias), 64*sizeof(float));
    ifs_conv_bias.close();
    
    // Golden Output
    ifstream ifs_golden_output("../bin/conv_output.bin", ios::in | ios::binary);
    ifs_golden_output.read((char*)(**conv_layer_golden_output_feature_map), 64*368*640*sizeof(float));    
    ifs_golden_output.close();
}

//--------------------------------------------------------------------------
// This is where fun begins.
//--------------------------------------------------------------------------
int main ()
{
    long double mse = 0.0;
    
    // Read reference inputs, parameters, and output
    read_bin_files();
   
    std::cout << "Beginning C model simulation..." << std::endl;
    
    model_conv (conv_layer_input_feature_map,
                conv_layer_weights,
                conv_layer_bias,
                conv_layer_output_feature_map
    );

    std::cout << "C model simulation complete!\n" << std::endl;
    
    // Compute Mean-Squared-Error
    for(int f = 0; f < 64; f++)
    {
        for(int i = 0; i < 368; i++)
        {
            for(int j = 0; j < 640; j++)
            {
                mse += std::pow((conv_layer_golden_output_feature_map[f][i][j] 
                                      - conv_layer_output_feature_map[f][i][j]), 2);
            }
        }
        
        #ifdef PRINT_DEBUG
            // Prints sample output values (first feature of each channel) for comparison
            // Modify as required for debugging
            int row = 0;
            int col = 0;
            
            cout << "Output feature[" << f << "][" << row << "][" << col << "]: ";
            cout << "Expected: " << conv_layer_golden_output_feature_map[f][row][col] << "\t"; 
            cout << "Actual: " << conv_layer_output_feature_map[f][row][col];
            cout << endl; 
        #endif
    }
    
    mse = mse / (64 * 368 * 640);

    std::cout << "Output MSE:  " << mse << std::endl;
    
    std::cout << "----------------------------------------" << std::endl;
    if(mse > 0 && mse < std::exp(-12))
    {
        std::cout << "Simulation SUCCESSFUL!!!" << std::endl;
    }
    else
    {
        std::cout << "Simulation FAILED :(" << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;

    return 0;
}

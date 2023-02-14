///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    sim.cpp
// Description: Test bench for tiling-based convolution 
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

fm_t  fixp_conv_layer_input_feature_map[3][736][1280];
wt_t  fixp_conv_layer_weights[64][3][7][7];
wt_t  fixp_conv_layer_bias[64];
fm_t  fixp_conv_layer_output_feature_map[64][368][640] = {0};

//--------------------------------------------------------------------------
// Read the reference files into test bench arrays
//--------------------------------------------------------------------------
void read_bin_files()
{
    // Input Feature Map
    ifstream ifs_conv_input("../bin/conv_input.bin", ios::in | ios::binary);
    ifs_conv_input.read((char*)(**conv_layer_input_feature_map), 3*736*1280*sizeof(float));
    ifs_conv_input.close();

    // Typecast to fixed-point 
    for(int c = 0; c < 3; c++)
        for(int i = 0; i < 736; i++)
            for(int j = 0; j < 1280; j++)
                fixp_conv_layer_input_feature_map[c][i][j] = (fm_t) conv_layer_input_feature_map[c][i][j];    
    
    // Weights
    ifstream ifs_conv_weights("../bin/conv_weights.bin", ios::in | ios::binary);
    ifs_conv_weights.read((char*)(***conv_layer_weights), 64*3*7*7*sizeof(float));
    ifs_conv_weights.close();
    
    // Typecast to fixed-point 
    for(int f = 0; f < 64; f++)
        for(int c = 0; c < 3; c++)
            for(int m = 0; m < 7; m++)
                for(int n =0; n < 7; n++)
                    fixp_conv_layer_weights[f][c][m][n] = (wt_t) conv_layer_weights[f][c][m][n];
    
    // Bias
    ifstream ifs_conv_bias("../bin/conv_bias.bin", ios::in | ios::binary);
    ifs_conv_bias.read((char*)(conv_layer_bias), 64*sizeof(float));
    ifs_conv_bias.close();
    
    // Typecast to fixed-point 
    for(int f = 0; f < 64; f++)
        fixp_conv_layer_bias[f] = (wt_t) conv_layer_bias[f];

    // Golden Output
    ifstream ifs_golden_output("../bin/conv_output.bin", ios::in | ios::binary);
    ifs_golden_output.read((char*)(**conv_layer_golden_output_feature_map), 64*368*640*sizeof(float));    
    ifs_golden_output.close();
}

//--------------------------------------------------------------------------
// This is where the real fun begins.
//--------------------------------------------------------------------------
int main ()
{
    long double mse = 0.0;
    
    // Read reference inputs, parameters, and output
    read_bin_files();
   
    std::cout << "Beginning HLS tiled-convolution simulation..." << std::endl;
    
    tiled_conv (fixp_conv_layer_input_feature_map,
                fixp_conv_layer_weights,
                fixp_conv_layer_bias,
                fixp_conv_layer_output_feature_map
    );
    
    std::cout << "Tiled-convolution simulation complete!\n" << std::endl;
    
    //Compute Mean-Squared-Error
    for(int f = 0; f < 64; f++)
    {
        for(int i = 0; i < 368; i++)
        {
            for(int j = 0; j < 640; j++)
            {
                mse += std::pow((conv_layer_golden_output_feature_map[f][i][j] 
                                 -(float) fixp_conv_layer_output_feature_map[f][i][j]), 2);
            }
        }
        #ifdef PRINT_DEBUG
            // Prints sample output values (first feature of each channel) for comparison
            // Modify as required for debugging
            int row = 0;
            int col = 0;
            
            cout << "Output feature[" << f << "][" << row << "][" << col << "]: ";
            cout << "Expected: " << conv_layer_golden_output_feature_map[f][row][col] << "\t"; 
            cout << "Actual: "   << fixp_conv_layer_output_feature_map[f][row][col];
            cout << endl; 
        #endif        
    }
    
    mse = mse / (64 * 368 * 640);

    std::cout << "\nOutput MSE:  " << mse << std::endl;
    
    std::cout << "----------------------------------------" << std::endl;
    #ifdef CSIM_DEBUG
        if(mse > 0 && mse < std::pow(10,-13))
        {
            std::cout << "Floating-point Simulation SUCCESSFUL!!!" << std::endl;
        }
        else
        {
            std::cout << "Floating-point Simulation FAILED :(" << std::endl;
        }
    #else
        if(mse > 0 && mse < std::pow(10,-3))
        {
            std::cout << "Fixed-point Simulation SUCCESSFUL!!!" << std::endl;
        }
        else
        {
            std::cout << "Fixed-point Simulation FAILED :(" << std::endl;
        }
    #endif
    std::cout << "----------------------------------------" << std::endl;

    return 0;
}

//--------------------------------------------------------------------------
// Test bench for your convolution codes.
//
// You should not need to modify this, except for debugging.
//
// Remove any print statements in your submission codes!
//--------------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <cmath>

#include "conv.h"

using namespace std;

//--------------------------------------------------------------------------
// Convolution layer inputs and reference output
//--------------------------------------------------------------------------
float   conv_layer_input_feature_map[64][184][320];
float   conv_layer_weights[64][64][3][3];
float   conv_layer_bias[64];
float   conv_layer_golden_output_feature_map[64][184][320];

fm_t    fixp_conv_layer_input_feature_map[64][184][320];
wt_t    fixp_conv_layer_weights[64][64][3][3];
wt_t    fixp_conv_layer_bias[64];

//--------------------------------------------------------------------------
// Computed outputs
//--------------------------------------------------------------------------
fm_t   fixp_conv_layer_output_feature_map[64][184][320] = {0};

//--------------------------------------------------------------------------
// Read the reference files into test bench arrays
//--------------------------------------------------------------------------
void read_bin_files()
{
    // Input Feature Map
    ifstream ifs_conv_input("../bin/conv_layer_input_feature_map.bin", ios::in | ios::binary);
    ifs_conv_input.read((char*)(**conv_layer_input_feature_map), 64*184*320*sizeof(float));
    ifs_conv_input.close();
    
    // Weights
    ifstream ifs_conv_weights("../bin/conv_layer_weights.bin", ios::in | ios::binary);
    ifs_conv_weights.read((char*)(***conv_layer_weights), 64*64*3*3*sizeof(float));
    ifs_conv_weights.close();
    
    // Bias
    ifstream ifs_conv_bias("../bin/conv_layer_bias.bin", ios::in | ios::binary);
    ifs_conv_bias.read((char*)(conv_layer_bias), 64*sizeof(float));
    ifs_conv_bias.close();

    // Golden Output
    ifstream ifs_golden_output("../bin/conv_layer_output_feature_map.bin", ios::in | ios::binary);
    ifs_golden_output.read((char*)(**conv_layer_golden_output_feature_map), 64*184*320*sizeof(float));    
    ifs_golden_output.close();
}

//--------------------------------------------------------------------------
// Convert the data types of every array element for specified 
// configuration.
//--------------------------------------------------------------------------
void convert_type()
{
    // Input Feature Map
    for(int c = 0; c < 64; c++)
        for(int i = 0; i < 184; i++)
            for(int j = 0; j < 320; j++)
                fixp_conv_layer_input_feature_map[c][i][j] = (fm_t) conv_layer_input_feature_map[c][i][j];
    
    // Weights
    for(int f = 0; f < 64; f++)
        for(int c = 0; c < 64; c++)
            for(int m = 0; m < 3; m++)
                for(int n =0; n < 3; n++)
                    fixp_conv_layer_weights[f][c][m][n] = (wt_t) conv_layer_weights[f][c][m][n];
    
    // Bias
    for(int f = 0; f < 64; f++)
        fixp_conv_layer_bias[f] = (wt_t) conv_layer_bias[f];
}

//--------------------------------------------------------------------------
// This is where fun begins.
//--------------------------------------------------------------------------
int main ()
{
    long double mse = 0.0;
    
    // Read reference inputs and output
    read_bin_files();
   
    // Convert to fixed-point types 
    convert_type();

    #ifdef CMODEL_SIM
        std::cout << "Beginning C model simulation..." << std::endl;
        model_conv (fixp_conv_layer_input_feature_map,
                    fixp_conv_layer_weights,
                    fixp_conv_layer_bias,
                    fixp_conv_layer_output_feature_map
        );
        std::cout << "C model simulation complete!\n" << std::endl;
    #else
        std::cout << "Beginning HLS tiled-convolution simulation..." << std::endl;
        tiled_conv (fixp_conv_layer_input_feature_map,
                    fixp_conv_layer_weights,
                    fixp_conv_layer_bias,
                    fixp_conv_layer_output_feature_map
        );
        std::cout << "Tiled-convolution simulation complete!\n" << std::endl;
    #endif
    
    //Compute MSE
    for(int f = 0; f < 64; f++)
    {
        for(int i = 0; i < 184; i++)
        {
            for(int j = 0; j < 320; j++)
            {
                mse += std::pow((conv_layer_golden_output_feature_map[f][i][j] 
                              - (float) fixp_conv_layer_output_feature_map[f][i][j]), 2);
            }
        }
        
        // Prints sample output values for comparison
        // Uncomment for debugging
        //cout << "Golden Output: " << conv_layer_golden_output_feature_map[f][0][0] << std::endl;
        //cout << "Actual Output: " << fixp_conv_layer_output_feature_map[f][0][0] << std::endl;
        //cout << std::endl;
    }
    
    mse = mse / (64 * 184 * 320);

    std::cout << "Output MSE:  " << mse << std::endl;
    
    std::cout << "----------------------------------------" << std::endl;
    #ifdef CSIM_DEBUG
        if(mse > 0 && mse < std::exp(-13))
        {
            std::cout << "Simulation SUCCESSFUL!!!" << std::endl;
        }
        else
        {
            std::cout << "Simulation FAILED :(" << std::endl;
        }
    #else
        if(mse > 0 && mse < std::exp(-3))
        {
            std::cout << "Simulation SUCCESSFUL!!!" << std::endl;
        }
        else
        {
            std::cout << "Simulation FAILED :(" << std::endl;
        }
    #endif
    std::cout << "----------------------------------------" << std::endl;

    return 0;
}

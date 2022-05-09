#include "conv.h"

//--------------------------------------------------------------------------
// Function to load input feature map tile block from DRAM to on-chip BRAM.
//
// TODO: This is a template function that you need to modify in Task B 
//       to handle border conditions. 
//--------------------------------------------------------------------------
void load_input_tile_block_from_DRAM (
    fm_t in_fm_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH], 
    fm_t in_fm[64][184][320], 
    int  ti, 
    int  tj, 
    int  d
)
{
    const int depth_offset  =  d * IN_BUF_DEPTH;
    const int height_offset = ti * OUT_BUF_HEIGHT; // OUT_BUF is intended, not a typo. 
    const int width_offset  = tj * OUT_BUF_WIDTH;
        
    INPUT_BUFFER_DEPTH:
    for(int c = 0; c < IN_BUF_DEPTH; c++)
    {
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < IN_BUF_HEIGHT; i++)
        {
            INPUT_BUFFER_WIDTH:
            for(int j = 0; j < IN_BUF_WIDTH; j++)
            {
                // TODO: Handle border features here
		        in_fm_buf[c][i][j] = in_fm[depth_offset + c][height_offset + i - 1][width_offset + j - 1];
            }
        }
    }
}

//--------------------------------------------------------------------------
// Function to load layer parameters (weights and bias) for convolution.
//
// You should not need to modify this function in Task B. 
//
// TODO: In Task C, you may add pragmas depending on your way 
//       of optimization.
//--------------------------------------------------------------------------
void load_layer_params_from_DRAM (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][3][3],
    wt_t bias_buf[OUT_BUF_DEPTH],
    wt_t weights[64][64][3][3],
    wt_t bias[64],
    int b,
    int d
)
{
    const int kernel_offset  = b * OUT_BUF_DEPTH;
    const int channel_offset = d * IN_BUF_DEPTH;

    WEIGHT_KERNEL_NUM:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        WEIGHT_KERNEL_DEPTH:
        for(int c = 0; c < IN_BUF_DEPTH; c++)
        {
            WEIGHT_KERNEL_HEIGHT:
            for(int kh = 0; kh < 3; kh++)
	        {
                WEIGHT_KERNEL_WIDTH:
	            for(int kw = 0; kw < 3; kw++)
	            {
	                weight_buf[f][c][kh][kw] = weights[kernel_offset + f][channel_offset + c][kh][kw];
                }
            }
        }
    }
    
    BIAS:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        bias_buf[f] = bias[kernel_offset + f];
    }

}

//------------------------------------------------------------------------------
// Function to save partial outputs on-chip for each input tile block processed.
//
// You should not need to modify this function in Task B. 
//
// TODO: In Task C, you may add pragmas depending on your way 
//       of optimization.
//------------------------------------------------------------------------------
void save_partial_output_tile_block (
    fm_t partial_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    wt_t bias_buf[OUT_BUF_DEPTH],
    int  d
)
{
    PARTIAL_OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        PARTIAL_OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < OUT_BUF_HEIGHT; i++)
        {
            PARTIAL_OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < OUT_BUF_WIDTH; j++)
            {
                if(d == 0) // Initialize buffer for first kernel group
                {
                    partial_out_buf[f][i][j]   = out_fm_buf[f][i][j] + bias_buf[f];
                }
                else // Accumulate otherwise
                {
                    partial_out_buf[f][i][j]  += out_fm_buf[f][i][j];
                }
            }
        }
    }
}

//--------------------------------------------------------------------------
// Function to store complete output tile block from BRAM to DRAM.
//
// You should not need to modify this function. 
//--------------------------------------------------------------------------
void store_output_tile_to_DRAM (
    fm_t out_fm[64][184][320], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    int  ti,
    int  tj,
    int  b
)
{
    const int depth_offset  =  b * OUT_BUF_DEPTH;
    const int height_offset = ti * OUT_BUF_HEIGHT;
    const int width_offset  = tj * OUT_BUF_WIDTH;

    OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < OUT_BUF_HEIGHT; i++)
        {
            OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < OUT_BUF_WIDTH; j++)
            {
                // ReLU in-place
                if(out_fm_buf[f][i][j] < (fm_t) 0)
                {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = (fm_t) 0;
                }
                else
                {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = out_fm_buf[f][i][j];
                }
            }
        }
    }
}

//---------------------------------------------------------------------------
// Perform synthesizable tiling-based convolution for entire ResNet-50 layer.
//---------------------------------------------------------------------------
#include "conv.h"

void tiled_conv (
    fm_t input_feature_map[64][184][320],
    wt_t layer_weights[64][64][3][3],
    wt_t layer_bias[64],
    fm_t output_feature_map[64][184][320]
)
{
//--------------------------------------------------------------------------
// Defines interface IO ports for HLS. 
// You should NOT modify these pragmas.
//--------------------------------------------------------------------------
#pragma HLS INTERFACE m_axi depth=64*184*320  port=input_feature_map   bundle=fm
#pragma HLS INTERFACE m_axi depth=64*64*3*3   port=layer_weights       bundle=wt
#pragma HLS INTERFACE m_axi depth=64          port=layer_bias          bundle=wt
#pragma HLS INTERFACE m_axi depth=64*184*320  port=output_feature_map  bundle=fm

#pragma HLS INTERFACE s_axilite register	port=return
    
    //--------------------------------------------------------------------------
    // On-chip buffers
    // You should NOT modify the buffer dimensions!
    //--------------------------------------------------------------------------
    fm_t conv_in_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    wt_t conv_wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][3][3];
    wt_t conv_bias_buf[OUT_BUF_DEPTH];
    fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];
    
    // Partial output storage buffer   
    fm_t partial_out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];
    
    // Process each tile iteratively
    for(int ti = 0; ti < N_TILE_ROWS; ti++)
    {
        for(int tj = 0; tj < N_TILE_COLS; tj++)
        {
            std::cout << "Processing Tile " << ti*N_TILE_COLS + tj + 1;
            std::cout << "/" << N_TILE_ROWS * N_TILE_COLS << std::endl;    

            //--------------------------------------------------------------------------
            // Your code for Task B and Task C goes here 
            //
            // Implement the required code to run convolution on an entire tile. 
            // Refer to utils.h for the required functions
            //
            // Hint: You may need to split the tile into tile blocks for processing
            //       as explained in the article.
            //--------------------------------------------------------------------------
        
        }
    }
}

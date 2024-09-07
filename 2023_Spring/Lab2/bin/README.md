## IMPORTANT INFORMATION

To run simulation in this lab, you need 4 binary files:
- *conv_input.bin* - Input 736 x 1280 (HD) RGB image
- *conv_weights.bin* - Convolution layer weights
- *conv_bias.bin* - Convolution layer bias
- *conv_output.bin* - Convolution layer output with 64 channels, each of dimension 368 x 640

However, GitHub does not allow uploading files larger than 25MB. :/ 

The problems that come with scale and large model inputs/outputs. Designs don't fit on board, compilers can't allocate memory on stack, and the list goes on and on. This is just the tip of the iceberg. 

Anyhoo, please download the binary files for this lab [here](https://www.dropbox.com/scl/fo/pxh1xy0bnmee465avowbe/AL47dPthXWXFsSAH7DlKJ3Y?rlkey=zipsumtzo0k8ccg4xhgmz1e4s&st=ezzlc72t&dl=0) and put these in your **PartA/bin** folder while cloning the repository. 

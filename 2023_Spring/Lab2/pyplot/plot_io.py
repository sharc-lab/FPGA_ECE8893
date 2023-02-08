import numpy as np
import matplotlib.pyplot as plt

# Read input feature map
X = np.fromfile('../bin/conv_input.bin', dtype='float32').reshape((3,736,1280))

# Normalize values for plotting RGB figure
X_norm = (X - X.min())/(X.max() - X.min())
plt.imshow(np.moveaxis(X_norm, 0, -1))
plt.axis('off')
plt.show()

# Read output feature map
Y = np.fromfile('../bin/conv_output.bin', dtype='float32').reshape((64,368,640))

# Plot each output channel as a subplot
f, axarr = plt.subplots(8,8)

for i in range(8):
    for j in range(8):
        idx = i*8 + j
        Z = Y[idx]
        Z_norm = (Z - Z.min())/(Z.max() - Z.min())
        
        axarr[i,j].imshow(Z_norm, cmap='Greys')
        axarr[i,j].axis('off')
        title = str(idx)
        axarr[i,j].set_title(title)
        
plt.show()

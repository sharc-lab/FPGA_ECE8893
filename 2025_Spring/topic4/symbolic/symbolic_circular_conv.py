import numpy as np 
import torch as t
import torch.nn.functional as F
import argparse

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="The neural part reference computation.")

    parser.add_argument(
        '--input_A_path',
        type=str,
        required=True,
        help='Path to the input .npy file (e.g., input1_A.npy)'
    )

    parser.add_argument(
        '--input_B_path',
        type=str,
        required=True,
        help='Path to the input .npy file (e.g., input1_B.npy)'
    )

    parser.add_argument(
        '--output_ref_path',
        type=str,
        required=True,
        help='Path to the reference output .npy file (e.g., output1_C.npy)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Computation device to use: "cpu" or "cuda:X" (e.g., "cuda:0")'
    )

    return parser.parse_args()

def binding_circular(A, B, alpha=1):
    """
    Binds two block codes vectors by blockwise cicular convolution. 

    Parameters
    ----------
    A: torch FloatTensor (_, _k, _l)
        input vector 1
    B: torch FloatTensor  (_, _k, _l)
        input vector 2
    alpha: int, optional
        specifies multiplicative factor for number of shifts (Default value '1')
    Returns
    -------
    C: torch FloatTensor (_k)
        k-dimensional offset vector that is the result of the binding operation.
    """
    ndim = A.dim()
    # add batch dimension (1) if not there yet
    if ndim==2: 
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
    
    batchSize,k,l = A.shape
    
    # prepare inputs
    A = t.unsqueeze(A,1) # input
    B = t.unsqueeze(B,2) # filter weigths
    B = t.flip(B, [3]) # flip input
    B = t.roll(B, 1, dims=3) # roll by one to fit addition

    # reshape for single CONV
    A = t.reshape(A, (1, A.shape[0]*A.shape[2], A.shape[3]))
    B = t.reshape(B, (B.shape[0]*B.shape[1], B.shape[2], B.shape[3]))

    # calculate C = t.remainder(B+A*alpha, self._L)
    C = F.conv1d(F.pad(A, pad=(0,l-1), mode='circular'), B, groups=k*batchSize)
    
    C = t.reshape(C, (batchSize, k, l))

    # Remove batch dimension if it was not there intially
    if ndim==2: 
        C = C.squeeze(0)
    return C

def mse_numpy(array1: np.ndarray, array2: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between two NumPy arrays.

    Parameters:
    - array1 (np.ndarray): First input array.
    - array2 (np.ndarray): Second input array.

    Returns:
    - float: The computed MSE value.

    Raises:
    - ValueError: If the input arrays do not have the same shape.
    """
    if array1.shape != array2.shape:
        raise ValueError(f"Shape mismatch: array1.shape = {array1.shape}, array2.shape = {array2.shape}")
    
    # Compute the squared differences
    squared_diff = (array1 - array2) ** 2
    
    # Compute the mean of the squared differences
    mse = np.mean(squared_diff)
    
    return mse

args = parse_arguments()
device = t.device(args.device if t.cuda.is_available() and args.device.startswith("cuda") else "cpu")
if args.device.startswith("cuda") and not t.cuda.is_available():
    print(f"CUDA is not available. Falling back to CPU.")
    device = t.device("cpu")

print(f"Using device: {device}")

array_A = np.load(args.input_A_path)
array_B = np.load(args.input_B_path)

tensor_A = t.tensor(array_A, dtype = t.float32)
tensor_B = t.tensor(array_B, dtype = t.float32)

tensor_A = tensor_A.to(device)
tensor_B = tensor_B.to(device)

tensor_C = binding_circular(tensor_A, tensor_B)

array_C = tensor_C.cpu().numpy()
ref_array_C = np.load(args.output_ref_path)

MSE_loss = mse_numpy(array_C, ref_array_C)
print("the MSE loss is:", MSE_loss)




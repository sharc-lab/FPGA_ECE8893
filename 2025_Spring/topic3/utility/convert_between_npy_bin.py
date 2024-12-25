#!/usr/bin/env python3

import argparse
import numpy as np

def npy_to_bin(input_npy_path, output_bin_path, dtype='float32'):
    """
    Convert an .npy file to a raw .bin file.
    
    Parameters
    ----------
    input_npy_path : str
        Path to the input .npy file.
    output_bin_path : str
        Path where the output .bin file will be saved.
    dtype : str, optional
        Data type to which the array will be cast before saving, by default 'float32'.
    """
    # 1. Load the .npy file
    arr = np.load(input_npy_path)
    
    
    # 2. Convert data type (optional)
    arr = arr.astype(dtype)
    
    # 3. Write to .bin (raw binary format)
    arr.tofile(output_bin_path)
    print(f"Successfully converted '{input_npy_path}' to '{output_bin_path}' as {dtype}.")


def bin_to_npy(input_bin_path, output_npy_path, shape=None, dtype='float32'):
    """
    Convert a raw .bin file to an .npy file.

    Parameters
    ----------
    input_bin_path : str
        Path to the input .bin file.
    output_npy_path : str
        Path where the output .npy file will be saved.
    shape : tuple of int, optional
        Shape to which the 1D data array should be reshaped.
        If None, the data remains 1D, by default None.
    dtype : str, optional
        Data type of the array in the .bin file, by default 'float32'.
    """
    # 1. Load the raw binary data as 1D array
    data = np.fromfile(input_bin_path, dtype=dtype)
    
    # 2. Reshape if shape is provided
    if shape is not None:
        data = data.reshape(shape)
    
    # 3. Save to .npy file
    np.save(output_npy_path, data)
    print(f"Successfully converted '{input_bin_path}' to '{output_npy_path}' as {dtype} with shape {data.shape}.")

def main():
    parser = argparse.ArgumentParser(description="Convert between .npy and .bin files.")
    parser.add_argument(
        "--mode",
        choices=["npy_to_bin", "bin_to_npy"],
        required=True,
        help="Choose conversion mode: 'npy_to_bin' or 'bin_to_npy'."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input file (.npy or .bin)."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output file (.bin or .npy)."
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Data type to use for reading/writing. Default is 'float32'."
    )
    parser.add_argument(
        "--shape",
        default=None,
        help="Shape for the array when converting from bin to npy (e.g. '100,200')."
    )

    args = parser.parse_args()

    if args.mode == "npy_to_bin":
        # Convert from .npy to .bin
        npy_to_bin(args.input, args.output, dtype=args.dtype)
    else:
        # Convert from .bin to .npy
        shape_tuple = None
        if args.shape is not None:
            # Convert shape string, e.g. "100,200" -> (100, 200)
            shape_tuple = tuple(map(int, args.shape.split(",")))
        bin_to_npy(args.input, args.output, shape=shape_tuple, dtype=args.dtype)

if __name__ == "__main__":
    main()



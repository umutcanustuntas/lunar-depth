import struct
import numpy as np
import cv2
import os
from pathlib import Path

MAX_UINT16 = 150


def read_pfm(path):
    """Read PFM file and return numpy array."""
    with open(path, 'rb') as pfm_file:
        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')

        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4
        
        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.reshape(decoded, shape) * scale

def convert_pfm_files():
    # Create output directories if they don't exist
    output_npy_dir = Path('pfm_output_npy')
    output_png_dir = Path('pfm_output_png')
    output_npy_dir.mkdir(exist_ok=True)
    output_png_dir.mkdir(exist_ok=True)

    # Get all PFM files from the pfm_files directory
    pfm_dir = Path('pfm_files')
    if not pfm_dir.exists():
        raise FileNotFoundError(f"Directory {pfm_dir} not found")

    for pfm_file in pfm_dir.glob('*.pfm'):
        print(f"Processing {pfm_file}")
        
        # Read PFM file
        depth_map = read_pfm(str(pfm_file))
        
        # Generate output filenames
        base_name = pfm_file.stem
        npy_path = output_npy_dir / f"{base_name}.npy"
        png_path = output_png_dir / f"{base_name}.png"

        # Create valid mask and filter values
        valid_mask = (depth_map < MAX_UINT16)
        depth_valid = depth_map[valid_mask]

        # Normalize to 0-1 for full depth map
        depth_min = np.min(depth_valid)
        depth_max = np.max(depth_valid)
        print(f"Original depth min: {depth_min}, max: {depth_max}")

        # Create normalized depth map with same shape as input
        depth_normalized = np.zeros_like(depth_map)
        depth_normalized[valid_mask] = (depth_map[valid_mask] - depth_min) / (depth_max - depth_min)
        
        # Save normalized depth to NPY
        np.save(str(npy_path), depth_normalized)
        
        # Convert to uint16 for PNG (0-65535)
        depth_uint16 = (depth_normalized * 65535).astype(np.uint16)
        cv2.imwrite(str(png_path), depth_uint16)

        print(f"Normalized depth min: {depth_normalized.min()}, max: {depth_normalized.max()}")
        print(f"PNG depth min: {depth_uint16.min()}, max: {depth_uint16.max()}")
        print(f"Saved: {npy_path} and {png_path}")
        

if __name__ == "__main__":
    try:
        convert_pfm_files()
        print("Conversion completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")




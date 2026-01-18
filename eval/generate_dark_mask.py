"""
Dark Mask Generator for Shadow Area Evaluation
"""

import cv2
import numpy as np
import os
import argparse


def generate_dark_masks(input_folder, output_folder, threshold_value=50, 
                        kernel_sizes=[3, 5, 7, 9], save_normal=False, prefix="", 
                        flat=False, no_kernel_suffix=False):
    
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    if not flat:
        for size in kernel_sizes:
            os.makedirs(os.path.join(output_folder, str(size)), exist_ok=True)
    
    kernels = {size: np.ones((size, size), np.uint8) for size in kernel_sizes}
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    total_files = len(png_files)
    
    if total_files == 0:
        print(f"No PNG files found in {input_folder}")
        return
    
    print(f"Found {total_files} PNG images to process...")
    print(f"Threshold value: {threshold_value}")
    print(f"Kernel sizes: {kernel_sizes}")
    print(f"Output folder: {output_folder}")
    print(f"Output mode: {'flat' if flat else 'subfolders'}")
    print("-" * 50)
    
    for idx, filename in enumerate(png_files):
        image_path = os.path.join(input_folder, filename)
        
        name_parts = filename.split('_')
        if len(name_parts) > 1:
            base_name = name_parts[-1].split('.')[0]
        else:
            base_name = filename.split('.')[0]
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: Could not read {filename}, skipping...")
            continue
        
        dark_mask = (image < threshold_value).astype(np.uint8) * 255
        
        if save_normal:
            normal_path = os.path.join(output_folder, f'normal_{filename}')
            cv2.imwrite(normal_path, dark_mask)
        
        for size in kernel_sizes:
            cleaned_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernels[size])
            
            if flat:
                if no_kernel_suffix:
                    output_name = f'cleaned_{filename}'
                else:
                    output_name = f'cleaned_{filename.split(".")[0]}_{size}.png'
                output_path = os.path.join(output_folder, output_name)
            else:
                output_name = f'{prefix}{base_name}.png'
                output_path = os.path.join(output_folder, str(size), output_name)
            
            cv2.imwrite(output_path, cleaned_mask)
        
        if (idx + 1) % 100 == 0 or (idx + 1) == total_files:
            print(f"Processed {idx + 1}/{total_files}: {filename}")
    
    print("-" * 50)
    print("All masks generated successfully!")
    print(f"Output saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Generate dark region masks for lunar images')
    
    parser.add_argument('--input_folder', type=str, required=True, help='Input images folder')
    parser.add_argument('--output_folder', type=str, required=True, help='Output masks folder')
    parser.add_argument('--threshold', type=int, default=50, help='Dark threshold 0-255 (default: 50)')
    parser.add_argument('--save_normal', action='store_true', help='Save uncleaned masks')
    parser.add_argument('--prefix', type=str, default="", help='Output filename prefix')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[3, 5, 7, 9], help='Kernel sizes (default: 3 5 7 9)')
    parser.add_argument('--flat', action='store_true', help='Save in flat folder instead of subfolders')
    parser.add_argument('--no_kernel_suffix', action='store_true', help='Do not add kernel size suffix in flat mode')
    
    args = parser.parse_args()
    
    generate_dark_masks(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        threshold_value=args.threshold,
        kernel_sizes=args.kernel_sizes,
        save_normal=args.save_normal,
        prefix=args.prefix,
        flat=args.flat,
        no_kernel_suffix=args.no_kernel_suffix
    )


if __name__ == "__main__":
    main()

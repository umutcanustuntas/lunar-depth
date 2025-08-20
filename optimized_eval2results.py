import os
import argparse
import numpy as np

from optimized_metrics import compute_metrics_parallel
from optimized_methods2evaluation import OptimizedDepthPreprocessor


def args_parser():
    parser = argparse.ArgumentParser(description="Optimized depth evaluation")
    parser.add_argument("gt_folder")
    parser.add_argument("preds_folder")
    parser.add_argument("--config_info", type=str, default="config_info")
    parser.add_argument("--absolute_depth", action="store_true")
    parser.add_argument("--relative_depth", action="store_true")
    parser.add_argument("--disparity", action="store_true")
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--max_gt_distance", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Shadow mask and labeling features
    parser.add_argument("--shadow_mask", type=str, help="Path to shadow mask directory")
    parser.add_argument("--labeling", type=str, help="Type of labeling to apply (e.g., obstacle, crater, mountain, ground)")
    parser.add_argument("--labeling_path", type=str, help="Directory containing the label png files for evaluation")
    
    # Distance range filtering - NOW WORKS FOR ALL FILE TYPES
    parser.add_argument("--distance_range", type=str, 
                       help="Distance range for evaluation (e.g., '30-60' for 30-60 meters, '100' for 0-100 meters). Works with all file types: .pfm, .png, .npy")
    
    return parser.parse_args()


def main():
    args = args_parser()
    
    # Initialize preprocessor
    preprocessor = OptimizedDepthPreprocessor(config_info=args.config_info, args=args)
    
    # Get file lists
    gt_files = sorted(os.listdir(args.gt_folder))
    pred_files = sorted(os.listdir(args.preds_folder))
    
    # Create full paths
    pred_paths = [os.path.join(args.preds_folder, f) for f in pred_files]
    gt_paths = [os.path.join(args.gt_folder, f) for f in gt_files]
    
    print(f"Processing {len(pred_paths)} image pairs with {args.num_workers} workers...")
    
    # Print masking information
    if args.shadow_mask:
        print(f"Using shadow mask from: {args.shadow_mask}")
    if args.labeling and args.labeling_path:
        print(f"Using labeling: {args.labeling} from: {args.labeling_path}")
    if args.distance_range:
        print(f"Using distance range: {args.distance_range}")
    
    # Parallel computation
    results, count = compute_metrics_parallel(
        pred_paths, gt_paths, preprocessor, 
        max_distance=args.max_gt_distance, 
        num_workers=args.num_workers,
        shadow_mask_dir=args.shadow_mask,
        labeling_type=args.labeling,
        labeling_path=args.labeling_path
    )
    
    if results is None:
        print("No valid results!")
        return
    
    # Print results
    print(f"\nResults ({count} valid files):")
    for metric_name, value in results.items():
        print(f"{metric_name}: {value:.4f}")


if __name__ == '__main__':
    main()

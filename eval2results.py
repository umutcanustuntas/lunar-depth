import os
import argparse
import numpy as np
from metrics import compute_metrics
from PIL import Image
from methods2evaluation import DepthPreprocessor

def args_parser():
    parser = argparse.ArgumentParser(prog="DepthEval",
                                     description="Evaluates depth estimation results")
    parser.add_argument("gt_folder")
    parser.add_argument("preds_folder")
    parser.add_argument("--config_info", type=str, default="config_info")
    parser.add_argument("--disparity", action="store_true")
    parser.add_argument("--absolute_depth", action="store_true")


    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--relative_depth", action="store_true")
    parser.add_argument("--resize", action="store_true")


    return parser.parse_args()



def main():
    args = args_parser()
    preprocessor = DepthPreprocessor(config_info = args.config_info,
                                     args=args)

    gt_depth_files = sorted(os.listdir(args.gt_folder))
    pred_depth_files = sorted(os.listdir(args.preds_folder))
    total = {
        "Abs Rel": 0.0,
        "Sq Rel": 0.0,
        "RMSE": 0.0,
        "RMSE Log": 0.0,
        "Log10": 0.0,
        "δ1": 0.0,
        "δ2": 0.0,
        "δ3": 0.0,
        "SI_log": 0.0,
        "F_A": 0.0
    }
    metrics_keys = total.keys()
    
    for pred_file, gt_file in zip(pred_depth_files, gt_depth_files):
        pred_path = os.path.join(args.preds_folder, pred_file)
        gt_path = os.path.join(args.gt_folder, gt_file)
        
        print(f"\nEvaluating {pred_file}:")
        print(f"GT: {gt_path}")

        processed_pred, processed_gt = preprocessor.process_depth(pred_path, gt_path)
        
        print("Processed shapes:", processed_pred.shape, processed_gt.shape)
        # Pass the dataset-specific absolute_depth flag into compute_metrics.
        result = compute_metrics(
            gt=processed_gt, 
            pred=processed_pred,
        )

        if result is None:
            continue
        
        for metric in metrics_keys:
            total[metric] += result[metric]

    for metric in metrics_keys:
        print(metric,": ", total[metric]/ len(pred_depth_files))

if __name__ == '__main__':
    main()

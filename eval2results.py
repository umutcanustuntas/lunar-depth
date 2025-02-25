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
    parser.add_argument("--max_gt_distance", type=int, default=100)


    return parser.parse_args()



def main():
    args = args_parser()
    preprocessor = DepthPreprocessor(config_info = args.config_info,
                                     args=args)

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
    
    scenes = ["moon1", "moon2", "moon3", "moon4", "moon5", "moon6", "moon7", "moon8", "moon9"]
    
    gt_depth_files = sorted(os.listdir(args.gt_folder))
    pred_depth_files = sorted(os.listdir(args.preds_folder))

    gt_lists =[]
    
    for i in scenes:
        temp = []
        for j in gt_depth_files:
            if i in j:
                temp.append(j)
        gt_lists.append(temp)

    pred_lists =[]
    
    for i in scenes:
        temp = []
        for j in pred_depth_files:
            if i in j:
                temp.append(j)
        pred_lists.append(temp)
        
    for pred_files, gt_files in zip(pred_lists, gt_lists):
        print(f"\nEvaluating {pred_files}:")

        temp = {
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
        for pred_file, gt_file in zip(pred_files, gt_files):

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
                temp[metric] += result[metric]
                total[metric] += result[metric]
        
        print("\nResults of", pred_files[0].split("_")[0])
        for metric in metrics_keys:
            temp[metric] /= len(pred_files)
            print(f"{metric}: {temp[metric]:.4f}")
        print("\n")

        input("Press Enter to continue...")


if __name__ == '__main__':
    main()

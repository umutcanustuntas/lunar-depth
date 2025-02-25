import os
import argparse
import numpy as np
from metrics import compute_metrics
from PIL import Image
import imageio.v3 as imageio
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
    parser.add_argument("--per_scene", action="store_true")

    #Arguments for masking and labeling
    parser.add_argument("--shadow_mask", type=str, help="Path to shadow mask directory")
    parser.add_argument("--labeling", type=str, help="Type of labeling to apply (e.g., OBSTACLE, CRATER, MOUNTAIN, GROUND)" ) #OBSTACLE, CRATER, MOUNTAIN, GROUND
    parser.add_argument("--labeling_path", type=str, help="Directory containing the label png files for evaluation" ) #Please enter the path of the label png files
    

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
    
    if args.per_scene:
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
                
                if args.shadow_mask:
                    # Get the base name without extension
                    base_name = os.path.splitext(pred_file)[0]
                    shadow_path = os.path.join(args.shadow_mask, f"cleaned_{base_name}_3.png")

                    if not os.path.exists(shadow_path):
                        #print(f"Shadow mask not found: {shadow_path}")
                        continue

                    #print(f"Using shadow mask: {shadow_path}")

                    # Load and apply shadow mask
                    shadow_img = Image.open(shadow_path)
                    shadow = np.array(shadow_img).astype(np.float32)
                    shadow_mask = shadow > 0  # Binary mask where shadow pixels are True

                    # Instead of indexing, use multiplication to preserve dimensions
                    processed_pred = processed_pred * shadow_mask
                    processed_gt = processed_gt * shadow_mask

                # Labeling Mask for Obstacle, Crater, Mountain
                if args.labeling: 
                    base_name = os.path.splitext(pred_file)[0]
                    labeling_path = os.path.join(args.labeling_path, f"{base_name}.png")

                    if not os.path.exists(labeling_path):
                        print(f"Labeling png file are not found: {labeling_path}")
                        continue



                    # Load and apply labeling mask labeling
                    labeling_img = imageio.imread(labeling_path)

                    if args.labeling.lower() == "obstacle":
                        target_color = np.array([232, 250, 80])
                    elif args.labeling.lower() == "crater":
                        target_color = np.array([120, 0, 200])
                    elif args.labeling.lower() == "mountain":
                        target_color = np.array([173, 69, 31])
                    elif args.labeling.lower() == "ground":
                        target_color = np.array([187, 70, 156])

                    # Create binary mask where True only for exact color matches
                    labeling_mask = np.all(labeling_img != target_color, axis=2)

                    processed_pred[labeling_mask]= 0
                    processed_gt[labeling_mask]= 0
            
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
    else:
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


    print("\nTotal results:")
    for metric in metrics_keys:
        print(metric,": ", total[metric]/ len(pred_depth_files))


if __name__ == '__main__':
    main()

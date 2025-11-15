import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import imageio.v3 as imageio


def compute_metrics(gt, pred, distance_mask=None):
    """Metrics computation"""
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: GT shape {gt.shape} and pred shape {pred.shape}")
    
    valid_mask = gt > 0
    
    # Apply distance mask if provided
    if distance_mask is not None:
        valid_mask = valid_mask & distance_mask
        
    min_valid_pixels = gt.shape[0] * gt.shape[1] * 0.001
    
    if valid_mask.sum() < min_valid_pixels:
        print("Warning: Too few valid pixels for reliable metrics")
        return None

    gt_valid = gt[valid_mask]
    pred_valid = pred[valid_mask]

    # Ensure positive values
    eps = 1e-6
    gt_valid = np.maximum(gt_valid, eps)
    pred_valid = np.maximum(pred_valid, eps)
    
    # Vectorized metric computations
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    sq_rel = np.mean(np.square(gt_valid - pred_valid) / gt_valid)
    rmse = np.sqrt(np.mean(np.square(gt_valid - pred_valid)))
    rmse_log = np.sqrt(np.mean(np.square(np.log(gt_valid) - np.log(pred_valid))))
    log10 = np.mean(np.abs(np.log10(gt_valid) - np.log10(pred_valid)))
    
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    delta1 = np.mean(thresh < 1.25)
    delta2 = np.mean(thresh < 1.25**2)
    delta3 = np.mean(thresh < 1.25**3)
    
    log_diff = np.log(pred_valid) - np.log(gt_valid)
    si_log = np.mean(log_diff**2) - np.mean(log_diff)**2
    
    f_a = np.mean(np.abs(gt_valid - pred_valid) < 0.5)
    
    return {
        "Abs Rel": abs_rel,
        "Sq Rel": sq_rel,
        "RMSE": rmse,
        "RMSE Log": rmse_log,
        "Log10": log10,
        "δ1": delta1,
        "δ2": delta2,
        "δ3": delta3,
        "SI_log": si_log,
        "F_A": f_a
    }


def apply_shadow_mask(pred, gt, pred_file, shadow_mask_dir):
    """Apply shadow mask to prediction and ground truth"""
    if not shadow_mask_dir:
        return pred, gt
    
    base_name = os.path.splitext(pred_file)[0]
    shadow_path = os.path.join(shadow_mask_dir, f"{base_name}.png")
    
    if not os.path.exists(shadow_path):
        return pred, gt
    
    # Load and apply shadow mask
    shadow_img = Image.open(shadow_path)
    shadow = np.array(shadow_img).astype(np.float32)
    shadow_mask = (shadow == 0)  
    
    # Apply mask
    pred = pred.copy()
    gt = gt.copy()
    pred[shadow_mask] = 0
    gt[shadow_mask] = 0
    
    return pred, gt


def apply_labeling_mask(pred, gt, pred_file, labeling_type, labeling_path):
    """Apply labeling mask to prediction and ground truth"""
    if not labeling_type or not labeling_path:
        return pred, gt
    
    base_name = os.path.splitext(pred_file)[0]
    label_file_path = os.path.join(labeling_path, f"{base_name}.png")
    
    if not os.path.exists(label_file_path):
        print(f"Labeling png file not found: {label_file_path}")
        return pred, gt
    
    # Color definitions
    OBSTACLE_COLOR = (232, 250, 80)
    CRATER_COLOR = (120, 0, 200)
    MOUNTAIN_COLOR = (173, 69, 31)
    GROUND_COLOR = (187, 70, 156)
    
    labeling_img = imageio.imread(label_file_path)
    
    # Select target color based on labeling type
    labeling_type_lower = labeling_type.lower()
    if labeling_type_lower == "obstacle":
        target_color = OBSTACLE_COLOR
    elif labeling_type_lower == "crater":
        target_color = CRATER_COLOR
    elif labeling_type_lower == "mountain":
        target_color = MOUNTAIN_COLOR
    elif labeling_type_lower == "ground":
        target_color = GROUND_COLOR
    else:
        print(f"Invalid labeling type: {labeling_type}")
        print("Valid types: obstacle, crater, mountain, ground")
        return pred, gt
    
    # Create labeling mask
    labeling_mask = np.all(labeling_img == target_color, axis=-1)
    labeling_mask = np.invert(labeling_mask)
    
    # Apply mask
    pred = pred.copy()
    gt = gt.copy()
    pred[labeling_mask] = 0
    gt[labeling_mask] = 0
    
    return pred, gt


def process_single_pair(args):
    """Process single depth pair for parallel execution with masking support"""
    (pred_path, gt_path, preprocessor, max_distance, 
     shadow_mask_dir, labeling_type, labeling_path) = args
    
    try:
        result = preprocessor.process_depth(pred_path, gt_path, max_distance)
        
        if len(result) == 3:
            pred, gt, distance_mask = result
        else:
            pred, gt = result
            distance_mask = None
            
        if pred is None or gt is None:
            return None
        
        # Get filename for mask processing
        pred_file = os.path.basename(pred_path)
        
        # Apply shadow mask if provided
        pred, gt = apply_shadow_mask(pred, gt, pred_file, shadow_mask_dir)
        
        # Apply labeling mask if provided
        pred, gt = apply_labeling_mask(pred, gt, pred_file, labeling_type, labeling_path)
        
        # Compute metrics with distance mask
        return compute_metrics(gt, pred, distance_mask)
        
    except Exception as e:
        print(f"Error processing {pred_path}: {e}")
        return None


def compute_metrics_parallel(pred_paths, gt_paths, preprocessor, max_distance=100, 
                           num_workers=4, shadow_mask_dir=None, labeling_type=None, 
                           labeling_path=None):
    """Parallel computation of metrics for multiple image pairs with masking support"""
    
    if num_workers == 1:
        # Sequential processing
        results = []
        for pred_path, gt_path in zip(pred_paths, gt_paths):
            result = process_single_pair((
                pred_path, gt_path, preprocessor, max_distance,
                shadow_mask_dir, labeling_type, labeling_path
            ))
            if result is not None:
                results.append(result)
    else:
        # Parallel processing
        args_list = [
            (pred_path, gt_path, preprocessor, max_distance,
             shadow_mask_dir, labeling_type, labeling_path)
            for pred_path, gt_path in zip(pred_paths, gt_paths)
        ]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_single_pair, args_list))
        
        results = [r for r in results if r is not None]
    
    if not results:
        print("No valid results!")
        return None, 0
    
    # Average all metrics
    metrics_names = list(results[0].keys())
    final_metrics = {}
    
    for metric_name in metrics_names:
        values = [r[metric_name] for r in results]
        final_metrics[metric_name] = np.mean(values)
    
    return final_metrics, len(results)

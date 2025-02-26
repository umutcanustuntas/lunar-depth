import numpy as np
from alignment import align_depth_least_square

def compute_metrics(gt, pred):
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: GT shape {gt.shape} and pred shape {pred.shape}")
    
    valid_mask = np.greater(gt, 0)
    print(valid_mask.sum(), "\n\n")
    min_valid_pixels = gt.shape[0] * gt.shape[1] * 0.001
    if valid_mask.sum() < min_valid_pixels:
        print("Warning: Too few valid pixels for reliable metrics")
        return None

    gt_valid = gt[valid_mask]
    pred_valid = pred[valid_mask]


    eps = 1e-6
    gt_valid = np.maximum(gt_valid, eps)
    pred_valid = np.maximum(pred_valid, eps)
    
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    sq_rel = np.mean(np.square(gt_valid - pred_valid) / gt_valid)
    rmse = np.sqrt(np.mean(np.square(gt_valid - pred_valid)))
    rmse_log = np.sqrt(np.mean(np.square(np.log(gt_valid) - np.log(pred_valid))))
    log10 = np.mean(np.abs(np.log10(gt_valid) - np.log10(pred_valid)))
    
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    print("THRESH: " , thresh)
    print("GT_Valid: " , gt_valid)
    print("PRED_Valid: " , pred_valid)
    delta1 = np.mean(thresh < 1.25)
    delta2 = np.mean(thresh < 1.25**2)
    delta3 = np.mean(thresh < 1.25**3)
    
    #print("THRESH: " , thresh)

    log_diff = np.log(pred_valid) - np.log(gt_valid)
    si_log = np.mean(log_diff**2) - np.mean(log_diff)**2
    
    f_a = np.mean(np.abs(gt_valid - pred_valid) < 0.5)
    
    print(f"Abs Rel: {abs_rel}")
    print(f"Sq Rel: {sq_rel}")
    print(f"RMSE: {rmse}")
    print(f"RMSE Log: {rmse_log}")
    print(f"Log10: {log10}" )
    print(f"δ1: {delta1}")
    print(f"δ2: {delta2}")
    print(f"δ3: {delta3}")
    print(f"SI_log: {si_log}")
    print(f"F_A: {f_a}")  
    
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

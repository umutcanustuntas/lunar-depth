import os
import yaml
import numpy as np
import cv2
from PIL import Image
from alignment import align_depth_least_square, disparity2depth, depth2disparity


class OptimizedDepthPreprocessor:
    def __init__(self, config_info="config_info", args=None):
        self.args = args
        path = os.path.join(os.path.dirname(__file__), 'configs', f'{config_info}.yaml')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.min_depth = float(config['min_depth'])
        self.max_depth = float(config['max_depth'])
        self.scale_factor = float(config['scale_factor'])
        
        # Parse distance range if provided
        self.distance_min, self.distance_max = self._parse_distance_range()
        self.distance_mask = None

    def _parse_distance_range(self):
        """Parse distance range from command line argument"""
        if not self.args or not hasattr(self.args, 'distance_range') or not self.args.distance_range:
            return None, None
        
        try:
            range_str = self.args.distance_range
            if '-' in range_str:
                min_dist, max_dist = range_str.split('-')
                return float(min_dist), float(max_dist)
            else:
                # Single value means max distance
                return 0.0, float(range_str)
        except:
            print(f"Warning: Invalid distance range format '{self.args.distance_range}'. Use format like '30-60' or '60'")
            return None, None

    def load_pfm(self, file_path):
        with open(file_path, 'rb') as f:
            header = f.readline().decode('utf-8').rstrip()
            if header not in ['PF', 'Pf']:
                raise ValueError('Not a PFM file.')
            
            dims = f.readline().decode('utf-8').strip()
            width, height = map(int, dims.split())
            scale = float(f.readline().decode('utf-8').strip())
            endian = '<' if scale < 0 else '>'
            
            data = np.fromfile(f, endian + 'f')
            shape = (height, width, 3) if header == 'PF' else (height, width)
            return np.reshape(data, shape)

    def load_depth(self, path, max_distance=450, is_gt=False):
        if path.endswith('.npy'):
            depth = np.load(path)
            if not is_gt and not self.args.absolute_depth:
                depth = depth * self.max_depth
        elif path.endswith('.png'):
            depth = np.array(Image.open(path)).astype(np.float32)
            if is_gt:
                depth = depth / self.scale_factor
        elif path.endswith('.pfm'):
            depth = self.load_pfm(path)
            # Apply max distance filtering for PFM
            depth = np.where(depth > max_distance, 0, depth)
            # Normalize if GT
            if is_gt and depth.max() > 0:
                depth = depth / depth.max()
                
        return depth

    def apply_distance_mask(self, depth, file_path):
        """Apply distance range mask to any depth map"""
        if self.distance_min is None or self.distance_max is None:
            return None
        
        # Create distance range mask
        distance_mask = (depth >= self.distance_min) & (depth <= self.distance_max) & (depth > 0)
        
        # Log information
        total_valid = (depth > 0).sum()
        masked_valid = distance_mask.sum()
        file_name = os.path.basename(file_path)
        
        print(f"Distance mask applied to {file_name}:")
        print(f"  Range: {self.distance_min}-{self.distance_max}m")
        print(f"  Valid pixels: {masked_valid}/{total_valid} ({100*masked_valid/max(total_valid,1):.1f}%)")
        
        return distance_mask

    def apply_median_scaling(self, pred, gt):
        mask = gt > 0
        if mask.sum() > 0:
            scale = np.median(gt[mask]) / np.median(pred[mask])
            return pred * scale
        return pred

    def process_depth(self, pred_path, gt_path=None, max_distance=100):
        # Load depths
        pred = self.load_depth(pred_path, is_gt=False)
        gt = self.load_depth(gt_path, max_distance=max_distance, is_gt=True)
        
        if pred is None or gt is None:
            raise ValueError("Either the prediction or ground truth depth map is None")

        # Squeeze and resize
        pred = np.squeeze(pred)
        gt = np.squeeze(gt)
        
        if self.args and self.args.resize:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        elif pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Alignment
        valid_mask = (gt > 0)
        
        if self.args and self.args.relative_depth:
            if self.args.disparity:
                gt_disparity, gt_non_neg_mask = disparity2depth(disparity=gt, return_mask=True)
                pred_non_neg_mask = pred > 0
                valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask
                
                disparity_pred, _, _ = align_depth_least_square(
                    gt_arr=gt_disparity, pred_arr=pred, valid_mask_arr=valid_nonnegative_mask)
                disparity_pred = np.clip(disparity_pred, a_min=1e-6, a_max=None)
                pred = disparity2depth(disparity_pred)
            else:
                pred, _, _ = align_depth_least_square(
                    gt_arr=gt, pred_arr=pred, valid_mask_arr=valid_mask)
        elif self.args and self.args.absolute_depth:
            pred = self.apply_median_scaling(pred, gt)

        # Clipping
        pred = np.clip(pred, a_min=self.min_depth, a_max=self.max_depth)
        pred = np.clip(pred, a_min=1e-6, a_max=None)
        
        # Apply distance range mask (for ALL file types now)
        distance_mask = self.apply_distance_mask(gt, gt_path)
        
        return pred, gt, distance_mask


# Backward compatibility
DepthPreprocessor = OptimizedDepthPreprocessor

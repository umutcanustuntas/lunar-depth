import os
import yaml
import numpy as np
import cv2
from PIL import Image
from scipy.optimize import least_squares
from alignment import align_depth_least_square, disparity2depth, depth2disparity


class ConfigLoader:
    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


class DepthPreprocessor:
    def __init__(self, config_info="config_info", args=None):
        print("Initializing DepthPreprocessor...")
        self.args = args
        path = os.path.join(os.path.dirname(__file__), 'configs', f'{config_info}.yaml')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        self.config_info = ConfigLoader.load_config(path)
        self.__init_parameters()

    def __init_parameters(self):
        # Dataset parameters
        print(self.config_info)
        self.min_depth = float(self.config_info['min_depth'])
        self.max_depth = float(self.config_info['max_depth'])
        self.scale_factor = float(self.config_info['scale_factor'])
        self.input_size = (
            int(self.config_info['input_width']),
            int(self.config_info['input_height'])
        )

        # Load crop parameters
        self.crop_params = self.config_info.get('crop_params', {})
        
    def load_pfm(self, file_path):
        with open(file_path, 'rb') as f:
            header = f.readline().decode('utf-8').rstrip()
            if header == 'PF':
                color = True
            elif header == 'Pf':
                color = False
            else:
                raise ValueError('Not a PFM file.')

            dims = f.readline().decode('utf-8').strip()
            width, height = map(int, dims.split())

            scale = float(f.readline().decode('utf-8').strip())
            endian = '<' if scale < 0 else '>'

            data = np.fromfile(f, endian + 'f')
            shape = (height, width, 3) if color else (height, width)

            return np.reshape(data, shape)
        
    def load_depth(self, path,max_distance=450, is_gt=False):
        print("Loading depth...")
        """Load depth with proper scaling.
        
        For GT images (png) the depth is divided by scale_factor.
        For .npy files, if not is_gt, we multiply by max_depth unless the
        --absolute_depth flag is set.
        """
        if path.endswith('.npy'):
            depth = np.load(path)
            if not is_gt and not self.args.absolute_depth:
                depth = depth * self.max_depth # if max depth is 1 nothing change, else it will be scaled to max_depth from 0-1
        elif path.endswith('.png'):
            depth = np.array(Image.open(path)).astype(np.float32)
            if is_gt:
                depth = depth / self.scale_factor # in png format some data is given more precise which is larger than real depth interval
        elif path.endswith('.pfm'):
            depth = self.load_pfm(path)
            print("depth min max:", depth.min(), depth.max())
            mask = depth > max_distance
            depth[mask] = 0
            if is_gt:
                depth = depth / depth.max()# in png format some data is given more precise which is larger than real depth interval

        return depth


    def apply_median_scaling(self, pred, gt):
        """Apply median scaling to prediction."""
        mask = gt > 0
        scale = np.median(gt[mask]) / np.median(pred[mask])
        return pred * scale


    def process_depth(self, pred_path, gt_path=None,max_distance=100):            
        # Load depths
        pred = self.load_depth(pred_path, is_gt=False)
        gt = self.load_depth(gt_path, max_distance=max_distance, is_gt=True)
        if pred is None or gt is None:
            raise ValueError("Either the prediction or ground truth depth map is None")

        # Squeeze to remove any extra singleton dimensions
        pred = np.squeeze(pred)
        gt = np.squeeze(gt)


        # Resize prediction to match GT if necessary
        if self.args and self.args.resize:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        elif pred.shape != gt.shape:
            print(f"WARNING: Prediction shape {pred.shape} does not match GT shape {gt.shape}")
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            print("Because of the warning, resizing prediction to match GT shape. \
                    In order to prevent this caution, please use --resize flag.")

        
        # Create valid mask (non-zero pixels in ground truth)
        valid_mask = (gt > 0)
        print("before alignment")
        print("gt min max ", gt.min(), gt.max())
        print("pred min max ", pred.min(), pred.max())
        
        # Check dataset configuration: relative or absolute depth
        if self.args and self.args.relative_depth:
            print("Performing disparity alignment...")
            # Perform least squares alignment in disparity space
            if self.args.disparity:
                gt_disparity, gt_non_neg_mask = disparity2depth(disparity=gt, return_mask=True)
                pred_non_neg_mask = pred > 0
                valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask
            
                disparity_pred, scale, shift = align_depth_least_square(
                    gt_arr=gt_disparity,
                    pred_arr=pred,
                    valid_mask_arr=valid_nonnegative_mask,
                    return_scale_shift=True,
                    max_resolution=None,
                )
    
                disparity_pred = np.clip(disparity_pred, a_min=1e-6, a_max=None)  # avoid 0 disparity 1e6 changed to 1e-3
                pred = disparity2depth(disparity_pred)

            else:
                print( "Disparity alignment skipped.")
                # Perform least squares alignment in depth space
                pred, scale, shift = align_depth_least_square(
                    gt_arr=gt,
                    pred_arr=pred,
                    valid_mask_arr=valid_mask,
                    return_scale_shift=True,
                    max_resolution=None,
                )


        elif self.args and self.args.absolute_depth:
            # For absolute depth (e.g., depthpro), apply median scaling
            if valid_mask.sum() > 0:
                pred = self.apply_median_scaling(pred, gt)
            else:
                print("Warning: No valid pixels found for median scaling.")

        else: # No alignment
            print("Alignment skipped.")

        print("after alignment")
        print("gt min max ", gt.min(), gt.max())
        print("pred min max ", pred.min(), pred.max())
        # Clip to min and max depth values
        pred = np.clip(pred, a_min=self.min_depth, a_max=self.max_depth)
        pred = np.clip(pred, a_min=1e-6, a_max=None)
        

        return pred, gt
    

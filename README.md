# Lunar Depth Estimation Evaluation


## Option with Parallel Processing

Fast evaluation with multiprocessing:

```bash
# Parallel Processing (4 workers)
python optimized_eval2results.py ground_truth/ predictions/ \
    --config_info config_info \
    --absolute_depth \
    --max_gt_distance 100 \
    --num_workers 4

# Evaluation with shadow mask
python optimized_eval2results.py ground_truth/ predictions/ \
    --config_info config_info \
    --absolute_depth \
    --num_workers 4 \
    --shadow_mask path/to/shadow/masks/

# Evaluation with labeling mask
python optimized_eval2results.py ground_truth/ predictions/ \
    --config_info config_info \
    --absolute_depth \
    --num_workers 4 \
    --labeling obstacle \
    --labeling_path path/to/labeling/masks/

# With both shadow and labeling masks
python optimized_eval2results.py ground_truth/ predictions/ \
    --config_info config_info \
    --absolute_depth \
    --num_workers 4 \
    --shadow_mask path/to/shadow/masks/ \
    --labeling crater \
    --labeling_path path/to/labeling/masks/

# Evaluate only 30-60 meter range (works with any file type) => For relative ground-truth please consider the interval itself is between 0 and 1 (LunarSim)
python optimized_eval2results.py ground_truth/ predictions/ \
    --config_info config_info \
    --absolute_depth \
    --distance_range "30-60" \
    --num_workers 4

# Evaluate only 0-100 meter range
python optimized_eval2results.py ground_truth/ predictions/ \
    --config_info config_info \
    --absolute_depth \
    --distance_range "100" \
    --num_workers 4

# Complete evaluation with all features
python optimized_eval2results.py ground_truth/ predictions/ \
    --config_info config_info \
    --absolute_depth \
    --resize \
    --max_gt_distance 100 \
    --num_workers 8 \
    --distance_range "30-60" \
    --shadow_mask path/to/shadow/masks/ \
    --labeling obstacle \
    --labeling_path path/to/labeling/masks/
```



### 1. Masking Types
- **Shadow Mask**: Filters shadow regions
- **Labeling Types**: 
  - `obstacle`: Obstacle regions
  - `crater`: Crater regions  
  - `mountain`: Mountain regions
  - `ground`: Ground regions

### 2. Parallel Processing 
- `num_workers > 1`: Parallel processing (fast)
- `num_workers = 1`: Sequential processing (low memory)
- Each image pair is processed in separate worker


### 3. Usage Notes
- Shadow mask files: Should be in `cleaned_{filename}_5.png` format
- Labeling files: Should be in `{filename}.png` format  
- `num_workers = 4-8` is generally optimal
- Memory usage increases with worker count


## Files Description

### Files:
- `optimized_eval2results.py`: **Main optimized evaluation script with parallel processing**
- `optimized_methods2evaluation.py`: Optimized depth preprocessing with numpy vectorization
- `optimized_metrics.py`: Parallel metrics computation with masking support

### Configuration:
- `config_info.yaml`: Evaluation configuration parameters

## Installation Requirements

```bash
#Create a new conda environment
conda create -n myenv python=3.9
conda activate myenv

#Install requirements
pip install -r Requirements.txt
```

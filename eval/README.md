# Evaluation Framework for Monocular Depth Estimation

This directory contains a comprehensive and flexible suite of Python scripts for evaluating the performance of monocular depth estimation models. The framework is optimized for parallel processing and provides detailed analysis options, including performance on specific semantic regions (e.g., obstacles, craters) and at various distance ranges.

---

## Key Features

*   **Multiple Input Formats:** Natively handles ground truth and prediction files in `.npy`, `.png`, and `.pfm` formats.
*   **Flexible Alignment Strategies:**
    *   **Absolute Depth:** Median scaling for models that predict metric depth (`--absolute_depth`).
    *   **Relative Depth:** Least-squares fitting for scale and shift for models that predict relative depth (`--relative_depth`).
    *   **Disparity Alignment:** Option to perform alignment in disparity space before converting to depth (`--disparity`).
*   **Comprehensive Metrics:** Computes a standard set of depth evaluation metrics:
    *   `Abs Rel`, `Sq Rel`, `RMSE`, `RMSE Log`, `Log10`, `δ1`, `δ2`, `δ3`, `SI_log`,
*   **Targeted Analysis with Masking:**
    *   **Semantic Masking:** Evaluate performance on specific object classes like `obstacle`, `crater`, `mountain`, or `ground` using RGB-encoded label maps (`--labeling` and `--labeling_path`).
    *   **Shadow Masking:** Exclude or isolate shadowed regions from the evaluation using binary shadow masks (`--shadow_mask`).
    *   **Distance Filtering:** Analyze performance within specific depth ranges, such as "30-60" meters or up to "100" meters (`--distance_range`). Addition to that, to evaluate filter relative data ranges like "0.3-0.6" could be used. 
*   **Efficient Processing:**
    *   **Parallel Execution:** Significantly speeds up evaluation on large datasets using multiple CPU cores (`--num_workers`).
    *   **Image Resizing:** Optional on-the-fly resizing of predictions to match ground truth dimensions (`--resize`).
*   **Utility Scripts:**
    *   Includes a `pfm2npy.py` script to convert `.pfm` files into `.npy` and normalized 16-bit `.png` files for easier use with other tools.

## Installation

It is recommended to use a virtual environment to manage dependencies. The `Requirements.txt` file should be located in the project's root directory.

1.  **Navigate to the project root and create a Conda environment:**
    ```bash
    # Assuming you are in the 'lunar-depth' root directory
    conda create -n lunarmde python=3.9 -y
    conda activate lunarmde
    ```

2.  **Install the required packages:**
    ```bash
    # From the root directory
    pip install -r requirements.txt
    ```

## How to Run the Evaluation

The main script for running evaluations is `eval2results.py`. It requires a path to the ground truth folder and a path to the predictions folder.

### Basic Command Structure

```bash
python eval2results.py /path/to/ground_truth /path/to/predictions [OPTIONS]
```

**Important:** The script assumes that the file names in the ground truth and prediction folders are sorted lexicographically and correspond to each other.

### Evaluation Examples

Below are some examples demonstrating how to use the script.

**1. Standard Absolute Depth Evaluation**
For models that output metric depth. This will use median scaling for alignment.

```bash
python eval2results.py /path/to/gt_folder /path/to/preds_folder \
    --absolute_depth \
    --num_workers 8
```

**2. Relative Depth Evaluation (from Disparity)**
For models that output inverse depth (disparity). This will use least-squares fitting for alignment in disparity space.

```bash
python eval2results.py /path/to/gt_folder /path/to/preds_folder \
    --relative_depth \
    --disparity \
    --num_workers 8
```

**3. Advanced: Evaluating Obstacles in a Specific Distance Range**
This example demonstrates how to combine masking features to get a fine-grained performance analysis. It will compute metrics only on pixels labeled as "obstacle" that fall between 10 and 50 meters in the ground truth.

```bash
python eval2results.py /path/to/gt_folder /path/to/preds_folder \
    --absolute_depth \
    --labeling obstacle \
    --labeling_path /path/to/your/label_masks \
    --distance_range "10-50" \
    --num_workers 8
```

### Dark Mask Generation Example
```bash
python generate_dark_mask.py \
    --input_folder ./data/LuSNAR/color0_tenth_main \
    --output_folder ./output/dark_mask \
    --flat --kernel_sizes 9 --save_normal --no_kernel_suffix
```

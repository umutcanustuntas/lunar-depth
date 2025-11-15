# Lunar Depth: A Framework for Monocular Depth Estimation

**Evaluating State-of-the-Art Monocular Depth Estimation for Lunar Rover Navigation.**

<p align="center">
  <a href="#multimedia">Multimedia</a> •
  <a href="#installation">Installation</a> •
  <a href="#citation">Paper</a> •
  <a href="https://github.com/your-username/LunarMDE/issues">Contact Us</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/your-paper-id"><img src="https://img.shields.io/badge/paper-coming%20soon-red.svg" alt="Paper-Link-Coming-Soon"></a>
  &nbsp;&nbsp;
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

This repository contains the official implementation of [Analysis of Monocular Depth Estimation for Lunar
Vehicles](https://arxiv.org/abs/your-paper-id), which systematically evaluates the performance of state-of-the-art Monocular Depth Estimation (MDE) networks for lunar rover missions. 
We provide a comprehensive framework for evaluating MDE models on both real-world data from the **Chang'e-3 mission** and simulation data from the
[**LunarSim**](https://github.com/PUTvision/LunarSim) and  [**LuSNAR**](https://github.com/zqyu9/LuSNAR-dataset) datasets.  


---

## Multimedia

<!-- Placeholder for a compelling GIF or image showing the system in action. 
     For example, a side-by-side of a rover camera view and the generated depth map. -->
<p align="center">
  <img src="path/to/your/awesome_promo.gif" alt="Lunar MDE in Action" width="80%"/>
</p>

---


## Key Features

*   **Comprehensive Evaluation Suite:** A flexible, evaluation suite (in the `eval` folder) to compute a wide range of depth estimation metrics.
*   **Targeted Analysis:** Built-in support for performance evaluation on specific lunar features like **obstacles, craters, mountains, ground,** and **shadowed regions**.
*   **Real & Simulated Data Support:** Tools and scripts for evaluating on simulated datasets (e.g., LunarSim) and real-world data.
*   **Chang'e-3 Tools:** Includes scripts for generating ground truth and evaluating MDE inferences on data from the Chang'e-3 mission.
*   **High Performance:** Optimized with parallel processing to rapidly evaluate large datasets.

## Installation

We recommend using `conda` to create a clean and isolated environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/LunarMDE.git
    cd LunarMDE
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n lunarmde python=3.9 -y
    conda activate lunarmde
    ```

3.  **Install the required packages:**
    The dependencies are listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

<!-- ## How to Run

This repository provides a powerful evaluation script to test MDE model predictions against ground truth data. The main script is `eval2results.py`.

### Alignment and Scaling Options

You must choose one of the following evaluation strategies:

*   `--absolute_depth`: For models that predict metric depth. Applies median scaling to align the prediction with the ground truth.
*   `--relative_depth`: For models that predict relative depth. Aligns the prediction to the ground truth using a least-squares fit for scale and shift.
*   `--disparity`: Use with `--relative_depth` if your model predicts disparity. Alignment is performed in disparity space before converting to depth.

### Evaluation Examples

*   **Basic Absolute Depth Evaluation:**
    ```bash
    python eval2results.py /path/to/ground_truth /path/to/predictions --absolute_depth
    ```

*   **Relative Depth Evaluation in Disparity Space:**
    ```bash
    python eval2results.py /path/to/ground_truth /path/to/predictions --relative_depth --disparity
    ```

*   **Advanced: Evaluate on Obstacles within a 30-60m range:**
    This example demonstrates how to combine filtering options.
    ```bash
    python eval2results.py /path/to/ground_truth /path/to/predictions \
        --absolute_depth \
        --labeling obstacle \
        --labeling_path /path/to/labeling_masks/ \
        --distance_range "30-60"
    ```

*   **Full Example with all features:**
    ```bash
    python eval2results.py /path/to/ground_truth /path/to/predictions \
        --config_info configs/config_info.yaml \
        --absolute_depth \
        --resize \
        --num_workers 8 \
        --distance_range "0-100" \
        --shadow_mask /path/to/shadow_masks/ \
        --labeling crater \
        --labeling_path /path/to/labeling_masks/
    ```

For more detailed instructions on running evaluations for specific datasets (like LunarSim or Chang'e-3), please see the documentation in their respective folders:
*   `./lunarsim/README.md`
*   `./change-3/README.md` -->

## Citation

If you use our work in your research, please consider citing our paper:

```bibtex
@article{your_name_2025_lunarmde,
  title={LunarMDE: A Framework for Monocular Depth Estimation on the Moon},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:your-paper-id},
  year={2025}
}
```

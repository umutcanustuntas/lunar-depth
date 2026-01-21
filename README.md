# Lunar Depth: Benchmarking of Monocular Depth Estimation Models for Lunar Rovers

**Evaluating State-of-the-Art Monocular Depth Estimation Models for Lunar Rover Navigation.**

<p align="center">
  <a href="#multimedia">Multimedia</a> •
  <a href="#installation">Installation</a> •
  <a href="#citation">Paper</a> •
  <a href="https://github.com/umutcanustuntas/lunar-depth/issues">Contact Us</a>
</p>

<p align="center">
  <a href="paper-link"><img src="https://img.shields.io/badge/paper-coming%20soon-red.svg" alt="Paper-Link-Coming-Soon"></a>
  &nbsp;&nbsp;
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

This repository contains the official implementation of **Benchmarking of Monocular Depth Estimation Models for Lunar Rovers,** <!--[Benchmarking of Monocular Depth Estimation Models for Lunar Rovers](paper-link ),--> which systematically evaluates the performance of state-of-the-art Monocular Depth Estimation (MDE) models for lunar rover missions. \
We provide a comprehensive framework for evaluating MDE models on both real-world data from the [Chang'e-3 mission](https://www.nssdc.ac.cn/nssdc_en/html/task/change3.html) and simulation data from the
[**LunarSim**](https://github.com/PUTvision/LunarSim) and  [**LuSNAR**](https://github.com/zqyu9/LuSNAR-dataset) datasets.  


---

## Multimedia


## 🎥 Demo Video
[![Demo Video](https://img.youtube.com/vi/G1rYp-F2MTE/hqdefault.jpg)](https://www.youtube.com/watch?v=G1rYp-F2MTE)


## Key Features

*   **Comprehensive Evaluation Suite:** A flexible, evaluation suite (in the `eval` folder) to compute a wide range of depth estimation metrics.
*   **Targeted Analysis:** Built-in support for performance evaluation on specific lunar features like **obstacles, craters, mountains, ground,** and **shadowed regions**, with performance analysis at different distance ranges.
*   **Real & Simulated Data Support:** Tools and scripts for evaluating on simulated datasets (e.g., LunarSim) and real-world data.
*   **Chang'e-3 Tools:** Includes scripts (in the `Chang'e-3` folder) for generating ground truth and evaluating MDE inferences on data from the Chang'e-3 mission.

## Installation

We recommend using `conda` to create a clean and isolated environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/umutcanustuntas/lunar-depth.git
    cd lunar-depth

    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n lunarmde python=3.9 -y
    conda activate lunarmde
    ```

3.  **Install the required packages:**
    The dependencies are listed in the `requirements.txt` file.
    ```bash
    pip install -r Requirements.txt
    ```



## Citation
Coming soon...
<!--
1. If you use our work in your research, please consider citing our paper:

```bibtex
@article{Aynalysis of Monocular Depth Estimation for Lunar
Vehicles,
  title={Analysis_of_Monocular_Depth_Estimation_for_Lunar_Vehicles},
  author={Aytac Sekmen1,♥, F. Emre Gunes1,♥, Furkan Horoz1,♥, H. Umut Isik1,♥, M. Alp Ozaydin1,♥, O. Altay
Topaloglu1,♥, Y. Alp Yeni1, S. Umutcan Ustundas1, H. Ersin Soken2, Erol Sahin1, R. Gokberk Cinbis1,♦, Sinan
Kalkan1,♦,},
  journal={arXiv preprint arXiv:paper-id},
  year={2025}
}
```
--> 












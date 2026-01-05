# [AAAI 2026] MotionPhysics: Learnable Motion Distillation for Text-Guided Simulation

### [[Project Page](https://wangmiaowei.github.io/MotionPhysics.github.io/)] [[arXiv](https://arxiv.org/abs/2601.00504v1)]

**Miaowei Wang**, **Jakub Zadrożny**, **Oisin Mac Aodha**, **Amir Vaxman**
School of Informatics, University of Edinburgh

![MotionPhysics](assets/teaser-1.png)

---

## Abstract

Accurately simulating existing 3D objects across a wide range of materials typically requires expert knowledge and extensive manual tuning of physical parameters. We present **MotionPhysics**, an end-to-end differentiable framework that infers plausible physical parameters directly from user-provided natural language prompts for a given 3D scene—without requiring ground-truth trajectories or annotated videos. Our approach first leverages a multimodal large language model to estimate material parameters constrained within physically plausible ranges. We then introduce a **learnable motion distillation loss** that extracts robust motion priors from pretrained video diffusion models, while minimizing appearance and geometry inductive biases to effectively guide the simulation process. We evaluate MotionPhysics on over 30 scenarios, including real-world, human-designed, and AI-generated 3D objects, spanning diverse materials such as elastic solids, metals, foams, sand, and both Newtonian and non-Newtonian fluids. Experimental results demonstrate that MotionPhysics produces visually realistic, text-guided dynamic simulations with automatically inferred and physically plausible parameters, outperforming existing state-of-the-art methods.

---

## Installation

We use the original **Gaussian Splatting** implementation as a submodule. Please clone this repository and install the dependencies as follows:

```bash
conda create -n MotionPhysics python=3.9
conda activate MotionPhysics

pip install -r requirements.txt
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install -e gaussian-splatting/submodules/simple-knn/
```

---

## Quick Start

Preprocessed Gaussian Splatting models are provided in the `./dataset` directory. The following commands will download the dataset, run a demo physics simulation, and save the results to the `./output` directory:

```bash
# Download the dataset from the anonymous link
gdown 1fE1iK_huQ1IXGXxEGzk7eCGUCarqHBkZ

# Unzip the dataset
unzip dataset.zip

# Run the physics test script
bash phys_test.sh
```

---

## Code Release

📌 **Note:** The complete source code and detailed usage guidelines will be released soon.

---

## Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@InProceedings{motionphysics2026,
  title     = {MotionPhysics: Learnable Motion Distillation for Text-Guided Simulation},
  author    = {Miaowei Wang and Jakub Zadrożny and Oisin Mac Aodha and Amir Vaxman},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

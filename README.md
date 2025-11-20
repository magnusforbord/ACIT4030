# ACIT4030 – NeRF Baseline & Hash Encoding Project

This repository contains the codebase for our ACIT4030 3D Graphics project.  
We implement two Neural Radiance Field (NeRF) variants:

- **Baseline NeRF** — classic MLP with positional encodings.  
- **Hash-Encoded NeRF** — a multi-resolution hash grid inspired by Instant-NGP.

Both models are trained on a synthetic “cow” dataset and evaluated using a custom interactive viewer with cached GPU previews, bilinear interpolation, and optional Gaussian blur.

---

## 📁 Repository Structure
├── ACIT4030_Baseline.ipynb       # Baseline NeRF training + viewer
├── ACIT4030_Hash.ipynb           # Hash-based NeRF training + viewer + metrics
├── nerf_model.py                 # NeRF + hash model implementations
├── train_nerf.py                 # Script version of baseline training
└── utils/
├── generate_cow_renders.py   # Synthetic cow data generation
├── helper_functions.py       # Loss, sampling, visualization helpers
└── plot_image_grid.py        # Utility for displaying grids of images

---

## 🚀 Running the Notebooks (Google Colab)

Both notebooks are designed to run in Google Colab with GPU enabled (T4).

### 1. Enable GPU
`Runtime → Change runtime type → Hardware accelerator: GPU`

### 2. Mount Drive & Install Dependencies  
The first cell:

- Mounts Google Drive
- Installs PyTorch3D + dependencies  
- Installs Gradio and Pillow for the viewer UI

### 3. Clone Repository  
The notebooks automatically clone this repo into `/content/acit4030-3d-project`.

### 4. Training Flow

Both notebooks follow the same pipeline:

1. Set global hyperparameters  
2. Generate synthetic cow dataset (camera poses, RGB, silhouettes)  
3. Build Monte-Carlo + full-grid ray samplers  
4. Instantiate the model (baseline or hash)  
5. Train using Huber loss on color + silhouette  
6. Save checkpoint + config (timestamped)  
7. Load model into the viewer  
8. Precompute GPU cache of ~168 views  
9. Launch interactive Gradio viewer

---

## 🧠 Models

### Baseline NeRF (`nerf_model.py`)
- Positional encoding for positions and view directions  
- MLP predicting density + color  
- Identical structure to classic NeRF papers/repositories  

### Hash-Encoded NeRF (`ACI030_Hash.ipynb`)
- Multi-resolution hash grid encoder  
- Trilinear interpolation of hashed feature corners  
- Harmonic embedding for view direction  
- Same rendering pipeline as baseline  
- Generally faster convergence and sharper edges, with some visible noise

---

## 🖼️ Interactive Viewer

Features:

- Mouse-drag orbit control  
- Real-time preview using cached GPU views  
- **Bilinear preview** (fastest)  
- **Blur preview** (smooths artifacts)  
- Full NeRF render triggered on mouse release  
- Automatic cache versioning using model hash  

---

## 📊 Performance Benchmarks

Both notebooks include a performance measurement cell that reports:

- Total parameter count  
- Bilinear preview latency (ms)  
- Blur preview latency (ms)  
- Full render time (ms)  
- GPU memory allocated & reserved  

These values are used to quantitatively compare baseline vs. hash NeRF.

---

## 🔧 Scripts

### `train_nerf.py`
A Python script version of the baseline training loop.  
Useful for non-Colab training or batch experiments.

---

## 🧩 Utilities

### `generate_cow_renders.py`
- Generates synthetic training data from a 3D cow mesh  
- Computes RGB images, silhouettes, and camera intrinsics/extrinsics  

### `helper_functions.py`
- Huber loss  
- Full-render visualization  
- Monte-Carlo sampling utilities  
- Rotation animation helper  

### `plot_image_grid.py`
- Simple grid visualization for debugging and comparison  

---

## 📝 Notes

- Colab GPU RAM is limited (~15GB).  
  We tuned:
  - number of training views  
  - rays per image  
  - points per ray  
  - render resolution  
  to stay within memory limits.

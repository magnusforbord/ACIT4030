# ACIT4030 – NeRF Baseline & Hash Encoding Project

This repository contains the codebase for our ACIT4030 3D Graphics project.  
We implement two Neural Radiance Field (NeRF) variants:

- **Baseline NeRF** — classic MLP with positional encodings.  
- **Hash-Encoded NeRF** — a multi-resolution hash grid inspired by Instant-NGP.

Both models are trained on a synthetic “cow” dataset and evaluated using a custom interactive viewer with cached GPU previews, bilinear interpolation, and optional Gaussian blur.
```text
├── ACIT4030_Baseline.ipynb        # Baseline NeRF training + interactive viewer
├── ACIT4030_Hash.ipynb            # Hash-based NeRF training + viewer + metrics
│
├── nerf_model.py                  # NeRF and hash-based NeRF implementations
├── train_nerf.py                  # Script version of baseline training
│
├── utils/
│   ├── generate_cow_renders.py    # Synthetic cow dataset generation
│   ├── helper_functions.py        # Loss functions, ray sampling, visualization helpers
│   └── plot_image_grid.py         # Utility for displaying grids of images
```
### Installing PyTorch3D from source (required on Google Colab)

If you are running this project on Google Colab, you must build PyTorch3D manually  
because no official wheels exist for Python 3.12 + CUDA on Colab.

Follow these steps:

```bash
# 1) Remove any preinstalled pytorch3d
!pip uninstall -y pytorch3d

# 2) Clone PyTorch3D at the 0.7.8 tag
!git clone https://github.com/facebookresearch/pytorch3d.git
%cd pytorch3d
!git checkout v0.7.8

# 3) Install dependencies and build the wheel
!pip install -q ".[all]"
```

The versions we used:
portalocker-3.2.0
iopath-0.1.10
tqdm-4.67.1
typing_extensions-4.15.0
pytorch3d-0.7.8

This works on Google Colab (T4 GPU) and takes about 10 minutes.
## Running the Notebooks (Google Colab)

Both notebooks are designed to run in Google Colab with GPU enabled (T4).

### 1. Enable GPU
`Runtime → Change runtime type → Hardware accelerator: GPU`

### 2. Mount Drive & Install Dependencies  
The first cell:

- Mounts Google Drive
- Installs PyTorch3D + dependencies  
- Installs Gradio and Pillow for the viewer UI

### 3. Clone Repository  
Cell 2 clones files from assignment 2 into `/content/acit4030-3d-project`.
Those files can also be found in this repo, but you need to change Cell 2, and adjust path if you would rather use these.

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

## Models

### Baseline NeRF (`ACIT4030_Baseline.ipynb`)
- Positional encoding for positions and view directions  
- MLP predicting density + color  
- Identical structure to classic NeRF papers/repositories  

### Hash-Encoded NeRF (`ACIT4030_Hash.ipynb`)
- Multi-resolution hash grid encoder  
- Trilinear interpolation of hashed feature corners  
- Harmonic embedding for view direction  
- Same rendering pipeline as baseline  
- Generally faster convergence and sharper edges, with some visible noise

---

## Interactive Viewer

Features:

- Mouse-drag orbit control  
- Real-time preview using cached GPU views  
- **Bilinear preview** (fastest)  
- **Blur preview** (smooths artifacts)  
- Full NeRF render triggered on mouse release  
- Automatic cache versioning using model hash  

---

## Performance Benchmarks

Both notebooks include a performance measurement cell that reports:

- Total parameter count  
- Bilinear preview latency (ms)  
- Blur preview latency (ms)  
- Full render time (ms)  
- GPU memory allocated & reserved  

These values are used to quantitatively compare baseline vs. hash NeRF.

---

## Scripts

### `train_nerf.py`
A Python script version of the baseline training loop.  
Useful for non-Colab training or batch experiments.

---

## Utilities

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

## Notes

- Colab GPU RAM is limited (~15GB).  
  We tuned:
  - number of training views  
  - rays per image  
  - points per ray  
  - render resolution  
  to stay within memory limits.

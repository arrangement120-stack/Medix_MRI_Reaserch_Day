# Medix: Interactive MRI Reconstruction Studio — Setup & Usage Tutorial

This guide explains how to launch, use, and export results from the **Medix: Physics-Guided Multi-Coil MRI Reconstruction Studio** across four supported environments:

| Platform | Hardware | Max Upload Size | Best For |
| :--- | :--- | :---: | :--- |
| **Hugging Face Spaces** *(Live Cloud)* | ZeroGPU (Shared) | `250 MB` | Zero-setup browser testing, single-slice `.h5`/`.npz` files, and the built-in synthetic 12-coil phantom |
| **Google Colab** | Free NVIDIA T4 GPU | `1024 MB` *(configurable)* | Large multi-slice NYU fastMRI `.h5` volumes with fast GPU inference |
| **Kaggle Notebooks** | Free NVIDIA T4 GPU | `1024 MB` *(configurable)* | Reproducible GPU sessions alongside Kaggle-hosted datasets |
| **Local Docker** | Local CPU or NVIDIA GPU | `1024 MB` *(configurable)* | Offline benchmarking, private scans, and full local control on `http://localhost:7860` |

> **Pretrained Weights:** All 7 deep learning checkpoints (`.pt`) are downloaded automatically on first run from Hugging Face Hub (`Mahmoud1213/Medix-MRI-weights`). You do not need to download weights manually.

---

## Option 1: Live Cloud Studio (Hugging Face Spaces)

No installation is required. Open the hosted application directly in your browser:

* **Live URL:** [https://huggingface.co/spaces/Mahmoud1213/Medix-MRI_demo](https://huggingface.co/spaces/Mahmoud1213/Medix-MRI_demo)

---

## Option 2: Run on Google Colab (Free T4 GPU)

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Enable GPU acceleration: **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**.
3. Run the following two cells:

**Cell 1 — Clone the repository and install dependencies:**
```python
!git clone https://github.com/arrangement120-stack/Medix_MRI_Reaserch_Day.git
%cd Medix_MRI_Reaserch_Day
!pip install -q -r requirements.txt
```

**Cell 2 — Launch Medix Studio with a public Gradio link:**
```python
!MAX_UPLOAD_MB=1024 GRADIO_SHARE=true python app.py
```

4. Wait until the cell output prints a public URL (`https://xxxx.gradio.live`) and click it to open the GPU-accelerated studio.

---

## Option 3: Run on Kaggle Notebooks (Free T4 GPU)

1. Open [Kaggle](https://www.kaggle.com/) and create a **New Notebook**.
2. In the right-hand **Notebook options** panel:
   * Set **Accelerator** to **GPU T4 x2** (or **GPU T4**).
   * Turn **Internet** **On**.
3. Run the following two cells:

**Cell 1 — Clone the repository and install dependencies:**
```python
!git clone https://github.com/arrangement120-stack/Medix_MRI_Reaserch_Day.git
%cd Medix_MRI_Reaserch_Day
!pip install -q -r requirements.txt
```

**Cell 2 — Launch Medix Studio with a public Gradio link:**
```python
!MAX_UPLOAD_MB=1024 GRADIO_SHARE=true python app.py
```

4. Click the printed `https://xxxx.gradio.live` link to open the interactive studio.

---

## Option 4: Run Locally with Docker (CPU or NVIDIA GPU)

### Prerequisites
* **Git** installed on your machine.
* **Docker**:
  * **Windows / macOS:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) running in the background.
  * **Linux / WSL 2 (Ubuntu):** Docker Engine or Docker Desktop with WSL integration enabled.
* *(Optional, for GPU mode)* An NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Verify Docker is available:
```bash
docker --version
```

### Step 1 — Clone the Repository
```bash
git clone https://github.com/arrangement120-stack/Medix_MRI_Reaserch_Day.git
cd Medix_MRI_Reaserch_Day
```

Confirm that `Dockerfile` (capital `D`), `.dockerignore`, `app.py`, `medix_engine.py`, and `requirements.txt` are present:
```bash
ls -la
```

### Step 2 — Build the Docker Image (One-Time Setup)
```bash
docker build -t medix-mri .
```
> *The first build downloads PyTorch and scientific dependencies and may take a few minutes.*

### Step 3 — Start the Container

**CPU Mode (Linux / macOS / WSL 2):**
```bash
docker run --rm -p 7860:7860 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  medix-mri
```

**CPU Mode (Windows PowerShell):**
```powershell
docker run --rm -p 7860:7860 `
  -v "$env:USERPROFILE\.cache\huggingface:/root/.cache/huggingface" `
  medix-mri
```

**NVIDIA GPU Mode (Linux / WSL 2 with Container Toolkit):**
```bash
docker run --rm --gpus all -p 7860:7860 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  medix-mri
```

> **Tip:** Mounting `~/.cache/huggingface` caches downloaded model weights on your host machine so they are not re-downloaded across container restarts. To raise the upload limit above the `1024 MB` Docker default (e.g., to `2048 MB`), add `-e MAX_UPLOAD_MB=2048`.

### Step 4 — Open in Browser
Wait for:
```text
Running on local URL:  http://0.0.0.0:7860
```
Then open **`http://localhost:7860`** in your browser. Press **`Ctrl + C`** in the terminal to stop the container.

---

## Using the Studio & Downloading Results

1. **Load Data:** Upload a raw NYU fastMRI multi-coil file (`.h5`), a preprocessed `.npz` slice, or click **Load Synthetic 12-Coil Brain Phantom (No Patient Data)** for an instant test. For multi-slice `.h5` files, select the desired slice index using the slider (only the selected slice is loaded into memory).
2. **Select Methods:** Choose one or more methods across 3 classical solvers (`Zero-Filled`, `CG-SENSE`, `TV-CS`) and 7 deep models (`Medix SSDU` at $T \in \{4, 6, 8\}$, `Unrolled-CG / MoDL`, `Medix K=0`, `fastMRI U-Net`, `fastMRI VarNet`).
3. **Configure Physics & Ablations (Optional):** Expand **3. Physics, Sampling & Ablation Controls** to adjust acceleration ($R \in \{4\times, 8\times\}$), mask type (`gaussian` or `equispaced`), ESPIRiT ACS size ($16, 24, 32$), CG steps ($K$), TV-CS FISTA iterations, error map gain, or toggle the **Coil-Support Mask** to suppress null-space background drift in Medix (SSDU).
4. **Run & Inspect:** Click **Run Reconstruction & Compare** to explore:
   * **Tab 1 — Quantitative Metrics & Export:** Live PSNR, SSIM, NMSE, NRMSE, HFEN, and latency leaderboard.
   * **Tab 2 — Interactive A vs. B Slider:** Drag-to-compare any two reconstructions plus $|A - B|$ difference map.
   * **Tab 3 — Unified Error Maps:** Side-by-side reconstructions and shared-scale absolute error maps.
   * **Tab 4 — Synchronized ROI Zoom:** Linked anatomical zoom boxes across all selected methods.
   * **Tab 5 — Physics Diagnostics:** Undersampled k-space, 1D sampling mask, dominant ESPIRiT coil map, and coil support.
   * **Tab 6 — Linear Algebra & CG:** Live adjointness error, spectral norm $\|A\|_2^2 \le 1$, learned $\mu_t$, HPD condition bounds $\kappa_t$, and inner CG Krylov residual decay.
5. **Download Results:**
   * **Metrics Table (`.csv`):** Download directly from **Tab 1** (and download the stage-by-stage linear-algebra diagnostics `.csv` from **Tab 6**).
   * **Full Results Bundle (`.zip`):** Download directly from **Tab 1**. Each archive contains:
     * `metrics.csv` — Complete quantitative benchmark table.
     * `run_info.json` — Exact physics, sampling, ablation, and runtime settings for reproducibility.
     * `figures/*.png` — High-resolution diagnostic figures (`metrics_chart`, `unified_error_grid`, `roi_comparison`, `physics_diagnostics`).
     * `reconstructions/*.png` and `reconstructions/*.npy` — 8-bit display images and full-precision `float32` arrays for every reconstructed method (excluding raw ground-truth patient scans for NYU fastMRI compliance).

---

## Troubleshooting

* **`Cannot connect to the Docker daemon`:** Start Docker Desktop (Windows/macOS) or run `sudo service docker start` on Linux.
* **`port is already allocated` (7860 in use):** Change the host port mapping to `-p 7861:7860` and open `http://localhost:7861`.
* **Docker cannot find `Dockerfile`:** Verify the filename starts with a capital `D` (`Dockerfile`).
* **Slow inference on Colab/Kaggle:** Confirm a T4 GPU accelerator is enabled in runtime settings before running Cell 2.

---

## Dataset Attribution

Data used in the preparation of this work were obtained from the **NYU fastMRI Initiative** database ([https://fastmri.med.nyu.edu](https://fastmri.med.nyu.edu)). NYU fastMRI investigators provided data but did not participate in the analysis or writing of this work. No patient scans from NYU fastMRI are redistributed in this repository; pretrained weights are shared strictly for non-commercial research and educational use.
# Medix: Interactive MRI Reconstruction Studio — Setup & Usage Tutorial

This guide explains how to launch, use, and export results from the **Medix: Physics-Guided Multi-Coil MRI Reconstruction Studio** across four supported environments:

| Platform | Hardware | Browser Upload Cap | Direct Disk / Drive Path | Best For |
| :--- | :--- | :---: | :---: | :--- |
| **Hugging Face Spaces** *(Live Cloud)* | ZeroGPU (Shared) | `250 MB` | Disabled *(cloud safety)* | Instant browser testing, single-slice `.h5`/`.npz` files, and the built-in synthetic 12-coil phantom |
| **Google Colab** | Free NVIDIA T4 GPU | Up to `10 GB` (`10240 MB`) | Enabled *(Google Drive)* | Multi-GB NYU fastMRI volumes (e.g., 1–4 GB multi-coil knee/brain scans) with zero upload wait |
| **Kaggle Notebooks** | Free NVIDIA T4 GPU | Up to `10 GB` (`10240 MB`) | Enabled *(`/kaggle/input`)* | Reproducible GPU sessions alongside Kaggle-hosted datasets |
| **Local Docker** | Local CPU or NVIDIA GPU | Up to `10 GB` (`10240 MB`) | Enabled *(mounted volumes)* | Offline benchmarking, private scans, and full local control on `http://localhost:7860` |

> **Pretrained Weights & Lazy Loading:** All 7 deep learning checkpoints (`.pt`) are downloaded automatically on first run from Hugging Face Hub (`Mahmoud1213/Medix-MRI-weights`). When reading multi-slice `.h5` files (even 1–10 GB volumes), `medix_engine.py` reads **only the selected slice** from disk (`hf["kspace"][slice_idx]`), keeping RAM usage low.

---

## Option 1: Live Cloud Studio (Hugging Face Spaces)

No installation is required. Open the hosted application directly in your browser:

* **Live URL:** [https://huggingface.co/spaces/Mahmoud1213/Medix-MRI_demo](https://huggingface.co/spaces/Mahmoud1213/Medix-MRI_demo)

*(For cloud stability, browser uploads on Hugging Face Spaces are capped at `250 MB`. If you have a multi-GB `.h5` volume and want to test a single slice on the live Space, see the single-slice extractor script in the **Tips for Large Multi-GB Files** section below.)*

---

## Option 2: Run on Google Colab (Free T4 GPU — Up to 10 GB)

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Enable GPU acceleration: **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**.
3. Run the following cells:

**Cell 1 — Clone the repository and install dependencies:**

```python
!git clone https://github.com/arrangement120-stack/Medix_MRI_Reaserch_Day.git
%cd Medix_MRI_Reaserch_Day
!pip install -q -r requirements.txt
```

**Cell 2 *(Optional, Recommended for 1–10 GB Files)* — Mount Google Drive:**

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 3 — Launch Medix Studio with 10 GB upload cap and public Gradio link:**

```python
!MAX_UPLOAD_MB=10240 GRADIO_SHARE=true python app.py
```

4. Click the printed public URL (`https://xxxx.gradio.live`) to open the GPU-accelerated studio.
5. **Fastest way to load 1–10 GB Drive files (No Browser Upload):**
   * Expand **Or Load Large File from Disk / Google Drive (No Size Limit)** in Section 1.
   * Paste the Drive path (for example: `/content/drive/MyDrive/fastmri_samples/knee/multicoil_val_file1001566.h5`).
   * Click **Load From Path (Lazy Single-Slice Read)**, choose your slice index, and run the benchmark.

---

## Option 3: Run on Kaggle Notebooks (Free T4 GPU — Up to 10 GB)

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

**Cell 2 — Launch Medix Studio (10 GB cap + Direct Disk Path enabled):**

```python
!MAX_UPLOAD_MB=10240 GRADIO_SHARE=true python app.py
```

4. Click the printed `https://xxxx.gradio.live` link. You can either upload files or load datasets directly from `/kaggle/input/...` using **Or Load Large File from Disk / Google Drive (No Size Limit)**.

---

## Option 4: Run Locally with Docker (CPU or NVIDIA GPU — Up to 10 GB)

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

### Step 3 — Start the Container (10 GB Upload Cap)

**CPU Mode (Linux / macOS / WSL 2):**

```bash
docker run --rm -p 7860:7860 \
  -e MAX_UPLOAD_MB=10240 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  medix-mri
```

**CPU Mode (Windows PowerShell):**

```powershell
docker run --rm -p 7860:7860 `
  -e MAX_UPLOAD_MB=10240 `
  -v "$env:USERPROFILE\.cache\huggingface:/root/.cache/huggingface" `
  medix-mri
```

**NVIDIA GPU Mode (Linux / WSL 2 with Container Toolkit):**

```bash
docker run --rm --gpus all -p 7860:7860 \
  -e MAX_UPLOAD_MB=10240 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  medix-mri
```

**Optional — Mount a Local MRI Folder for Instant Zero-Upload Loading:**

If your `.h5` files are already on disk (e.g., `/path/to/mri_data`), mount that folder into `/data`:

```bash
docker run --rm -p 7860:7860 \
  -e MAX_UPLOAD_MB=10240 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/mri_data:/data:ro \
  medix-mri
```

Then enter `/data/your_scan.h5` inside **Or Load Large File from Disk / Google Drive (No Size Limit)** in the web UI.

> **Tip:** Mounting `~/.cache/huggingface` caches downloaded model weights on your host machine so they are not re-downloaded across container restarts.

### Step 4 — Open in Browser

Wait for:

```text
Running on local URL:  http://0.0.0.0:7860
```

Then open **`http://localhost:7860`** in your browser. Press **`Ctrl + C`** in the terminal to stop the container.

---

## Tips for Large Multi-GB Files (1–10 GB)

* **Prefer Direct Disk / Drive Path over Browser Upload:** Even with `MAX_UPLOAD_MB=10240`, uploading a 4 GB file through a browser tab can be slow or time out over public tunnels. Mounting Google Drive (Colab) or a local folder (Docker) and clicking **Load From Path** reads a single slice in under a second.
* **Extract a Single Slice for Hugging Face Spaces (`250 MB` limit):** Run this short snippet in Python/Colab to save one central slice (~25–35 MB) from any multi-GB `.h5` file:

```python
import h5py

src_path = "/content/drive/MyDrive/fastmri_samples/knee/multicoil_val_file1001566.h5"
out_path = "/content/single_slice.h5"

with h5py.File(src_path, "r") as src, h5py.File(out_path, "w") as dst:
    mid = src["kspace"].shape[0] // 2
    dst.create_dataset("kspace", data=src["kspace"][mid : mid + 1])
    if "reconstruction_rss" in src:
        dst.create_dataset("reconstruction_rss", data=src["reconstruction_rss"][mid : mid + 1])
    for k, v in src.attrs.items():
        dst.attrs[k] = v
```

---

## Using the Studio & Downloading Results

1. **Load Data:** Upload a raw NYU fastMRI multi-coil file (`.h5`), a preprocessed `.npz` slice, load a direct path from Disk/Drive (Colab/Kaggle/Docker), or click **Load Synthetic 12-Coil Brain Phantom (No Patient Data)**. For multi-slice `.h5` files, select the desired slice index using the slider.
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
* **Browser upload timeout on 1–10 GB files:** Use **Or Load Large File from Disk / Google Drive (No Size Limit)** instead of uploading through the browser.
* **Slow inference on Colab/Kaggle:** Confirm a T4 GPU accelerator is enabled in runtime settings before running the launch cell.

---

## Dataset Attribution

Data used in the preparation of this work were obtained from the **NYU fastMRI Initiative** database ([https://fastmri.med.nyu.edu](https://fastmri.med.nyu.edu)). NYU fastMRI investigators provided data but did not participate in the analysis or writing of this work. No patient scans from NYU fastMRI are redistributed in this repository; pretrained weights are shared strictly for non-commercial research and educational use.
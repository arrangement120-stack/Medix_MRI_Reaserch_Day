# Medix: Interactive MRI Reconstruction Studio — Setup & Usage Tutorial

This guide explains how to launch, use, automate, and export results from the **Medix: Physics-Guided Multi-Coil MRI Reconstruction Studio** across six supported workflows:

| Platform / Mode | Hardware | File Size Limit | Direct Disk / Drive Path | Best For |
| :--- | :--- | :---: | :---: | :--- |
| **Hugging Face Spaces** *(Live Cloud)* | ZeroGPU (Shared) | `250 MB` *(cloud safety)* | Disabled | Instant browser testing, single-slice `.h5`/`.npz` files, and the built-in synthetic 12-coil phantom |
| **Google Colab** | Free NVIDIA T4 GPU | **No Limit** *(disk-bound)* | Enabled *(Google Drive)* | Multi-GB NYU fastMRI volumes (e.g., 1–4+ GB knee/brain scans) with zero upload wait |
| **Kaggle Notebooks** | Free NVIDIA T4 GPU | **No Limit** *(disk-bound)* | Enabled *(`/kaggle/input`)* | Reproducible GPU sessions alongside Kaggle-hosted datasets |
| **Local Docker** | Local CPU or NVIDIA GPU | **No Limit** *(disk-bound)* | Enabled *(mounted folders)* | Offline benchmarking, private scans, and full local control on `http://localhost:7860` |
| **Programmatic API** (`gradio_client`) | Remote HF ZeroGPU or Local Docker | `250 MB` *(HF)* / **No Limit** *(Local)* | Supported on Local/Colab | Calling Medix from Python pipelines, scripts, or external web apps |
| **Direct Terminal CLI** (`medix_terminal_cli.py`) | Local CPU/GPU, Colab, Kaggle, Docker | **No Limit** | Native local paths | Headless batch benchmarking and multi-GB scans with zero browser overhead |

> **Pretrained Weights & Lazy Single-Slice Loading:** All 7 deep learning checkpoints (`.pt`) are downloaded automatically on first run from Hugging Face Hub (`Mahmoud1213/Medix-MRI-weights`). When reading multi-slice `.h5` files (even multi-GB volumes), `medix_engine.py` reads **only the selected slice** from disk (`hf["kspace"][slice_idx]`), keeping RAM usage low.

---

## Option 1: Live Cloud Studio (Hugging Face Spaces)

No installation is required. Open the hosted application directly in your browser:

* **Live URL:** [https://huggingface.co/spaces/Mahmoud1213/Medix-MRI_demo](https://huggingface.co/spaces/Mahmoud1213/Medix-MRI_demo)

*(For cloud stability, uploads on Hugging Face Spaces are capped at `250 MB`. To test a single slice from a multi-GB `.h5` volume on the live Space, use the extractor snippet under **Tips for Large Multi-GB Volumes** below.)*

---

## Option 2: Run on Google Colab (Free T4 GPU — No Size Limit)

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Enable GPU acceleration: **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**.
3. Run the following cells:

**Cell 1 — Clone the repository and install dependencies:**

```python
!git clone https://github.com/arrangement120-stack/Medix_MRI_Reaserch_Day.git
%cd Medix_MRI_Reaserch_Day
!pip install -q -r requirements.txt
```

**Cell 2 *(Recommended for Multi-GB Files)* — Mount Google Drive:**

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 3 — Launch Medix Studio:**

```python
!python app.py
```

*(Colab is detected automatically and a public `https://xxxx.gradio.live` link is generated without extra flags.)*

4. Click the printed `gradio.live` link to open the GPU-accelerated studio.
5. **Fastest way to load 1+ GB Google Drive files (No Browser Upload):**
   * In Section 1, use the **Load Directly from Disk / Google Drive (Recommended for 1+ GB Volumes)** box.
   * Paste your Drive file path (for example: `/content/drive/MyDrive/fastmri_samples/knee/multicoil_val_file1001566.h5`).
   * Click **Load From Path (Lazy Single-Slice Read)**, choose your slice index, and click **Run Reconstruction & Compare**.

---

## Option 3: Run on Kaggle Notebooks (Free T4 GPU — No Size Limit)

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

**Cell 2 — Launch Medix Studio:**

```python
!python app.py
```

*(Kaggle is detected automatically and prints a public `https://xxxx.gradio.live` link.)*

4. Click the printed link. You can upload files directly or paste any `/kaggle/input/...` path into **Load Directly from Disk / Google Drive (Recommended for 1+ GB Volumes)**.

---

## Option 4: Run Locally with Docker (CPU or NVIDIA GPU — No Size Limit)

### Prerequisites

* **Git** installed on your machine.
* **Docker**:
  * **Windows / macOS:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) running in the background.
  * **Linux / WSL 2 (Ubuntu):** Docker Engine or Docker Desktop with WSL integration enabled.
* *(Optional, for GPU mode)* An NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Verify Docker is ready:

```bash
docker --version
```

### Step 1 — Clone the Repository

```bash
git clone https://github.com/arrangement120-stack/Medix_MRI_Reaserch_Day.git
cd Medix_MRI_Reaserch_Day
```

Confirm that `Dockerfile` (capital `D`), `.dockerignore`, `app.py`, `medix_engine.py`, `medix_terminal_cli.py`, and `requirements.txt` are present:

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

**Optional — Mount a Local MRI Folder for Instant Zero-Upload Loading:**

If your `.h5` files are already on disk (e.g., `/path/to/mri_data`), mount that folder into `/data`:

```bash
docker run --rm -p 7860:7860 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/mri_data:/data:ro \
  medix-mri
```

Then enter `/data/your_scan.h5` inside **Load Directly from Disk / Google Drive (Recommended for 1+ GB Volumes)** in the web UI.

> **Tip:** Mounting `~/.cache/huggingface` persists downloaded model weights on your host machine so they are not re-downloaded when restarting the container.

### Step 4 — Open in Browser

Wait until the terminal prints:

```text
Running on local URL:  http://0.0.0.0:7860
```

Then open **`http://localhost:7860`** in your browser. Press **`Ctrl + C`** in the terminal to stop the container.

---

## Option 5: Programmatic API Access (`gradio_client`)

Both the live Hugging Face Space and the local/Colab/Kaggle Gradio server expose clean API endpoints:
* **`/reconstruct`** — Runs multi-method reconstruction and returns metrics, `.csv`, and the `.zip` results bundle.
* **`/la_diagnostics`** — Runs adjointness, spectral norm $\|A\|_2^2 \le 1$, $\mu_t / \kappa_t$, and CG Krylov residual diagnostics.

Install the lightweight client once:

```bash
pip install gradio_client
```

### Example 1 — Call the Live Hugging Face Space (ZeroGPU)

```python
from gradio_client import Client, handle_file

# Optional: pass hf_token="hf_..." for higher ZeroGPU quota / private Spaces
client = Client("Mahmoud1213/Medix-MRI_demo")

result = client.predict(
    file_obj=handle_file("single_slice.h5"),
    slice_idx=0,
    selected_methods=["Zero-Filled (ZF)", "CG-SENSE (ESPIRiT)", "Medix (SSDU - T=8)"],
    accel=4,
    mask_type="gaussian",
    acs_size=24,
    cg_steps_medix=6,
    tv_iters=40,
    apply_support_mask=False,
    err_gain=10,
    roi_x=35,
    roi_y=35,
    roi_size=64,
    api_name="/reconstruct",
)

csv_path = result[3]  # Metrics leaderboard (.csv)
zip_path = result[4]  # Full results archive (.zip)
print("Metrics CSV:", csv_path)
print("Results ZIP:", zip_path)
```

### Example 2 — Call Local Docker or Colab/Kaggle Server (Direct Disk Path)

```python
from gradio_client import Client

# Replace with your printed https://xxxx.gradio.live URL if calling Colab/Kaggle
client = Client("http://localhost:7860")

result = client.predict(
    file_obj=None,
    local_path_str="/data/multicoil_val_file1001566.h5",
    slice_idx=18,
    selected_methods=["Zero-Filled (ZF)", "Medix (SSDU - T=8)", "Unrolled-CG / MoDL (Sup UB)"],
    accel=4,
    mask_type="gaussian",
    acs_size=24,
    cg_steps_medix=6,
    tv_iters=40,
    apply_support_mask=False,
    err_gain=10,
    roi_x=35,
    roi_y=35,
    roi_size=64,
    api_name="/reconstruct",
)

print("Metrics CSV:", result[3])
print("Results ZIP:", result[4])
```

*(You can also click **Use via API** in the bottom footer of the web UI to view auto-generated Python, JavaScript, and `cURL` snippets.)*

---

## Option 6: Direct Terminal CLI (`medix_terminal_cli.py`)

Run reconstructions directly from the command line without launching the Gradio web UI—ideal for batch scripts, WSL/Linux terminals, Colab/Kaggle, and multi-GB `.h5` volumes.

### Step 1 — Clone & Enter the Repository Folder

**Local Terminal / WSL Ubuntu:**

```bash
git clone https://github.com/arrangement120-stack/Medix_MRI_Reaserch_Day.git
cd Medix_MRI_Reaserch_Day
pip install -q -r requirements.txt
```

**Google Colab / Kaggle:**

* If running inside a **notebook cell**, prefix shell commands with `!` and use `%cd`:
  ```python
  !git clone https://github.com/arrangement120-stack/Medix_MRI_Reaserch_Day.git
  %cd Medix_MRI_Reaserch_Day
  !pip install -q -r requirements.txt
  ```
* If using the **Colab Terminal pane**, note that it opens in `/content` by default (separate from your notebook cells). Always navigate into the cloned repository first:
  ```bash
  cd /content/Medix_MRI_Reaserch_Day
  ```

### Step 2 *(Optional)* — Add Your Hugging Face API Token (`HF_TOKEN`)

Setting a free Hugging Face Read token ([create one here](https://huggingface.co/settings/tokens)) prevents anonymous download rate limits and speeds up fetching the pretrained weights from `Mahmoud1213/Medix-MRI-weights`.

* **Linux / macOS / WSL / Colab Terminal:**
  ```bash
  export HF_TOKEN="hf_your_token_here"
  ```
* **Colab / Kaggle Notebook Cell:**
  ```python
  %env HF_TOKEN=hf_your_token_here
  ```
* **Windows PowerShell:**
  ```powershell
  $env:HF_TOKEN="hf_your_token_here"
  ```

### Step 3 — Quick Sanity Test (Built-In 12-Coil Phantom)

Verify the CLI and model downloads work before loading a `.h5` scan:

```bash
python medix_terminal_cli.py --methods core --save-plots --out ./phantom_results
```

*(In a Colab/Kaggle notebook cell, prefix with `!`, i.e., `!python medix_terminal_cli.py ...`.)*

### Step 4 — Locate Your `.h5` Scan Path

Make sure you have the exact path to your `.h5` file:

* **Google Colab (after mounting Google Drive):**
  ```bash
  find /content/drive/MyDrive -name "*.h5"
  ```
* **Kaggle:**
  ```bash
  find /kaggle/input -name "*.h5"
  ```

### Step 5 — Run Reconstruction on Your `.h5` Scan

**Recommended (using the `core` shortcut):**

```bash
python medix_terminal_cli.py \
  --input "/path/to/your_scan.h5" \
  --slice 18 \
  --methods core \
  --accel 4 \
  --mask gaussian \
  --acs 24 \
  --cg-steps 6 \
  --save-plots \
  --out ./slice18_results
```

**Or select specific methods manually *(names are case-sensitive)*:**

```bash
python medix_terminal_cli.py \
  --input "/path/to/your_scan.h5" \
  --slice 18 \
  --methods "Zero-Filled (ZF)" "CG-SENSE (ESPIRiT)" "Medix (SSDU - T=8)" \
  --accel 4 \
  --mask gaussian \
  --acs 24 \
  --cg-steps 6 \
  --save-plots \
  --out ./slice18_results
```

> **CLI Tips:**
> * Always wrap `--input` in quotes if the file path contains spaces.
> * Omit `--slice` to automatically reconstruct the central slice.
> * **Method Shortcuts:** `core`, `classical`, or `all`.
> * **Valid Exact Method Names *(case-sensitive)*:** `Zero-Filled (ZF)`, `CG-SENSE (ESPIRiT)`, `TV-CS (FISTA)`, `Medix (SSDU - T=8)`, `Medix (SSDU - T=6)`, `Medix (SSDU - T=4)`, `Unrolled-CG / MoDL (Sup UB)`, `Medix K=0 (Sup, No CG)`, `fastMRI U-Net (Sup)`, `fastMRI VarNet (Sup)`.

### Step 6 — Inspect Saved Results

Every CLI run writes the following outputs to the `--out` directory:

* `metrics.csv` — Quantitative leaderboard (PSNR, SSIM, NMSE, NRMSE, HFEN, latency).
* `run_info.json` — Exact physics, sampling, and runtime configuration.
* `reconstructions/*.png` & `reconstructions/*.npy` — Display images and `float32` arrays.
* `figures/*.png` — Diagnostic comparison plots (saved when `--save-plots` is passed).

### Run the CLI Inside Docker (No Web Server Started)

Ensure `medix_terminal_cli.py` is included in the `COPY` line of your `Dockerfile` (`COPY app.py medix_engine.py medix_terminal_cli.py ./`), rebuild once, and run (include `-e HF_TOKEN=...` optionally):

```bash
docker run --rm \
  -e HF_TOKEN="hf_your_token_here" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/mri_data:/data \
  medix-mri \
  python medix_terminal_cli.py --input /data/scan.h5 --methods core --save-plots --out /data/medix_out
```

---

## Tips for Large Multi-GB Volumes

* **Prefer Direct Disk / Drive Path or CLI over Browser Upload:** Although Colab, Kaggle, and Docker impose no file size cap, uploading multi-GB files through a browser tab can be slow or time out over public tunnels. Mounting Google Drive (Colab), using `/kaggle/input` (Kaggle), mounting a local folder (Docker), or running `medix_terminal_cli.py` reads a single slice in under a second.
* **Extract a Single Slice for Hugging Face Spaces (`250 MB` cloud limit):** Run this short script in Python or Colab to save one central slice (~25–35 MB) from any multi-GB `.h5` file:

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

1. **Load Data:** Upload a raw NYU fastMRI multi-coil file (`.h5`), a preprocessed `.npz` slice, load a direct path from Disk/Drive (Colab/Kaggle/Docker), or click **Load Synthetic 12-Coil Brain Phantom (No Patient Data)**. For multi-slice `.h5` files, choose the slice index with the slider.
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
* **`port is already allocated` (7860 in use):** Map another host port, e.g., `-p 7861:7860`, and open `http://localhost:7861`.
* **Docker cannot find `Dockerfile`:** Verify the filename starts with a capital `D` (`Dockerfile`).
* **Browser upload timeout on multi-GB files:** Use **Load Directly from Disk / Google Drive (Recommended for 1+ GB Volumes)** or `medix_terminal_cli.py` instead of uploading through the browser.
* **Slow inference on Colab/Kaggle:** Confirm a T4 GPU accelerator is enabled in runtime settings before running `!python app.py` or `python medix_terminal_cli.py`.
* **`No such file or directory` for `medix_terminal_cli.py` in Colab Terminal:** Run `cd /content/Medix_MRI_Reaserch_Day` first—the Colab Terminal opens in `/content` separately from notebook cells.
* **`ValueError: Unknown method(s)` in CLI:** Method names are case-sensitive (for example, `CG-SENSE (ESPIRiT)` uses a lowercase `i`). Use `--methods core`, `classical`, or `all` as shortcuts.

---

## Dataset Attribution

Data used in the preparation of this work were obtained from the **NYU fastMRI Initiative** database ([https://fastmri.med.nyu.edu](https://fastmri.med.nyu.edu)). NYU fastMRI investigators provided data but did not participate in the analysis or writing of this work. No patient scans from NYU fastMRI are redistributed in this repository; pretrained weights are shared strictly for non-commercial research and educational use.

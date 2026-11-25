Save the Markdown below as **`DOCKER_SETUP.md`** (or paste it directly into your repository's **`README.md`**). Replace `<YOUR-GITHUB-REPO-URL>` with your GitHub repository link before committing.

```markdown
# Running Medix MRI Reconstruction Studio Locally with Docker

The **Medix Interactive MRI Reconstruction and Benchmarking Platform** is available in two modes:

| Mode | Access / Setup | Max Upload Size | Best For |
| :--- | :--- | :---: | :--- |
| **Live Cloud Demo** | [Hugging Face Spaces](https://huggingface.co/spaces/Mahmoud1213/Medix-MRI_demo) (ZeroGPU) | `250 MB` | Instant browser testing, single-slice `.h5`/`.npz` files, and the built-in synthetic 12-coil phantom |
| **Local Docker** | Runs on `http://localhost:7860` (CPU or NVIDIA GPU) | `1024 MB` *(configurable)* | Large multi-slice NYU fastMRI `.h5` volumes, offline benchmarking, and private scans |

---

## Prerequisites

* **Git** installed on your machine
* **Docker**:
  * **Windows / macOS:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) (running in the background)
  * **Linux / WSL 2:** Docker Engine or Docker Desktop with WSL integration enabled
* *(Optional, for GPU acceleration)* An NVIDIA GPU with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed

Verify Docker is ready before continuing:

```bash
docker --version

```

---

## Step 1: Clone the Repository

Clone from GitHub (or directly from the Hugging Face Space):

```bash
# Option A: Clone from GitHub
git clone <YOUR-GITHUB-REPO-URL>
cd <YOUR-REPO-FOLDER>

# Option B: Clone directly from Hugging Face Spaces
git clone [https://huggingface.co/spaces/Mahmoud1213/Medix-MRI_demo](https://huggingface.co/spaces/Mahmoud1213/Medix-MRI_demo)
cd Medix-MRI_demo

```

Confirm that `Dockerfile`, `.dockerignore`, `app.py`, `medix_engine.py`, and `requirements.txt` are present in the directory:

```bash
ls -la

```

---

## Step 2: Build the Docker Image

Run the build command from inside the repository folder (only required once, or after modifying the code):

```bash
docker build -t medix-mri .

```

> **Note:** The first build downloads PyTorch and scientific Python dependencies and may take several minutes depending on your internet connection.

---

## Step 3: Run the Container

### CPU Mode (Works on Any Machine)

**Linux / macOS / WSL 2 (Ubuntu):**

```bash
docker run --rm -p 7860:7860 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  medix-mri

```

**Windows (PowerShell):**

```powershell
docker run --rm -p 7860:7860 `
  -v "$env:USERPROFILE\.cache\huggingface:/root/.cache/huggingface" `
  medix-mri

```

### NVIDIA GPU Mode (Faster Inference)

Add `--gpus all` to accelerate all unrolled and baseline models on a local CUDA GPU:

```bash
docker run --rm --gpus all -p 7860:7860 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  medix-mri

```

> **Why mount `~/.cache/huggingface`?** Pretrained Medix and fastMRI model weights (`.pt`) are downloaded automatically on the first run. Mounting this cache folder persists the weights on your host machine so they are not re-downloaded when restarting the container.

### Customizing the Upload Size Limit (Optional)

Docker defaults to `MAX_UPLOAD_MB=1024` (1 GB). To raise or lower this cap (for example, to `2048` MB for very large volumes), pass `-e MAX_UPLOAD_MB=2048`:

```bash
docker run --rm -p 7860:7860 \
  -e MAX_UPLOAD_MB=2048 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  medix-mri

```

---

## Step 4: Open and Use the Studio

1. Wait until the terminal displays:
```text
Running on local URL:  [http://0.0.0.0:7860](http://0.0.0.0:7860)

```


2. Open your web browser and navigate to **`http://localhost:7860`**.
3. Run a quick verification test:
* Click **Load Synthetic 12-Coil Brain Phantom** (or upload an NYU fastMRI `.h5` / `.npz` file).
* Select one or more reconstruction methods (e.g., **Core 4**).
* Adjust physics, sampling, or ablation settings under **3. Physics, Sampling & Ablation Controls** if desired.
* Click **Run Reconstruction & Compare** to inspect metrics, export `.csv` tables, compare reconstructions via the A/B slider, view synchronized ROI zooms and unified error maps, or run live Linear-Algebra & Krylov checks.



To stop the server at any time, press **`Ctrl + C`** in the terminal.

---

## Quick Troubleshooting

* **`Cannot connect to the Docker daemon`**: Start Docker Desktop (Windows/macOS) or the Docker service (`sudo service docker start` on Linux), then retry.
* **`port is already allocated` (7860 in use)**: Map a different host port, e.g., `-p 7861:7860`, and open `http://localhost:7861`.
* **Build cannot find `Dockerfile**`: Ensure the file is named `Dockerfile` with a capital `D` (Linux filesystems are case-sensitive).

---

## Dataset Attribution

Data used in the preparation of this work were obtained from the **NYU fastMRI Initiative** database (`https://fastmri.med.nyu.edu`). NYU fastMRI investigators provided data but did not participate in the analysis or writing of this work. No patient scans from NYU fastMRI are redistributed in this repository; pretrained weights are shared strictly for non-commercial research and educational use.

```

```

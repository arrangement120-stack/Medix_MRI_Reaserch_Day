import io
import inspect
import math
import os
import re
import subprocess
import sys
import time
from typing import Dict, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from scipy.ndimage import gaussian_laplace
from skimage.metrics import structural_similarity as sk_ssim

try:
    import fastmri
except ImportError:
    env = os.environ.copy()
    env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FASTMRI"] = "0.3.0"
    env["VCS_VERSIONING_PRETEND_VERSION_FOR_FASTMRI"] = "0.3.0"
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "install", "-q", "--no-deps",
            "https://github.com/facebookresearch/fastMRI/archive/refs/heads/main.zip",
        ],
        env=env,
    )
    import fastmri

from fastmri import evaluate as fm_eval
from fastmri.data import transforms as T
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.models import Unet as FastMRIUnet, VarNet as FastMRIVarNet


MODEL_REPO = "Mahmoud1213/Medix-MRI-weights"

CLASSICAL_METHODS = [
    "Zero-Filled (ZF)",
    "CG-SENSE (ESPIRiT)",
    "TV-CS (FISTA)",
]

DEEP_METHODS = {
    "Medix (SSDU - T=8)": ("best_medix_ssdu_T8.pt", "medix", {"T": 8, "K": 6}),
    "Medix (SSDU - T=6)": ("best_medix_ssdu_T6.pt", "medix", {"T": 6, "K": 6}),
    "Medix (SSDU - T=4)": ("best_medix_ssdu_T4.pt", "medix", {"T": 4, "K": 6}),
    "Unrolled-CG / MoDL (Sup UB)": ("best_unrolled_cg_modl_sup.pt", "medix", {"T": 8, "K": 6}),
    "Medix K=0 (Sup, No CG)": ("best_medix_k0_no_cg.pt", "medix", {"T": 8, "K": 0}),
    "fastMRI U-Net (Sup)": ("best_fastmri_unet.pt", "unet", {}),
    "fastMRI VarNet (Sup)": ("best_fastmri_varnet.pt", "varnet", {}),
}

ALL_METHODS = CLASSICAL_METHODS + list(DEEP_METHODS.keys())

_MODEL_CACHE: Dict[str, nn.Module] = {}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 1. EXACT MRI PHYSICS & PREPROCESSING
# ==============================================================================
def fft2c_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_complex(fastmri.fft2c(torch.view_as_real(x.contiguous())).contiguous())


def ifft2c_complex(k: torch.Tensor) -> torch.Tensor:
    return torch.view_as_complex(fastmri.ifft2c(torch.view_as_real(k.contiguous())).contiguous())


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(x.contiguous()).permute(0, 3, 1, 2).contiguous()


def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())


def safe_center_crop_or_pad(data: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    Ht, Wt = int(target_shape[0]), int(target_shape[1])
    H, W = data.shape[-2], data.shape[-1]
    pad_h, pad_w = max(0, Ht - H), max(0, Wt - W)
    if pad_h > 0 or pad_w > 0:
        data = F.pad(data, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
    return T.center_crop(data, (Ht, Wt))


def remove_readout_oversampling_np(kspace_np: np.ndarray, target_h: int = 320) -> np.ndarray:
    H = kspace_np.shape[-2]
    if H <= target_h:
        return kspace_np.astype(np.complex64, copy=False)
    img_ro = np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(kspace_np, axes=(-2,)), axis=-2, norm="ortho"),
        axes=(-2,),
    )
    h0 = (H - target_h) // 2
    img_crop = img_ro[..., h0 : h0 + target_h, :]
    k_crop = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(img_crop, axes=(-2,)), axis=-2, norm="ortho"),
        axes=(-2,),
    )
    return k_crop.astype(np.complex64)


def svd_coil_compress_volume_np(kspace_vol: np.ndarray, max_coils: int = 12):
    S, C, H, W = kspace_vol.shape
    if C <= max_coils:
        return kspace_vol.astype(np.complex64, copy=False), 1.0
    ah, aw = min(48, H), min(48, W)
    h0, w0 = (H - ah) // 2, (W - aw) // 2
    cal = kspace_vol[:, :, h0 : h0 + ah, w0 : w0 + aw]
    cal_mat = np.transpose(cal, (1, 0, 2, 3)).reshape(C, -1).astype(np.complex64)
    cov = cal_mat @ cal_mat.conj().T
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    V_k = eigvecs[:, order[:max_coils]]
    retained = float(eigvals[order[:max_coils]].sum() / (eigvals.sum() + 1e-12))
    k_comp = np.einsum("ck,schw->skhw", V_k.conj(), kspace_vol, optimize=True).astype(np.complex64)
    return k_comp, retained


def generate_cartesian_mask_1d(
    W: int, accel: int, center_fraction: float, mask_type: str, seed: int = 42
):
    num_low_freqs = max(8, int(round(W * center_fraction)))
    pad = (W - num_low_freqs + 1) // 2
    if mask_type == "gaussian":
        rng = np.random.RandomState(seed)
        mask = np.zeros(W, dtype=np.float32)
        mask[pad : pad + num_low_freqs] = 1.0
        target_total = max(num_low_freqs + 2, int(round(W / float(accel))))
        n_outer = max(1, target_total - int(mask.sum()))
        outer_idx = np.flatnonzero(mask == 0.0)
        coords = (outer_idx - (W / 2.0)) / (W / 2.0)
        probs = np.exp(-0.5 * (coords / 0.45) ** 2)
        probs = probs / (probs.sum() + 1e-12)
        n_pick = min(n_outer, len(outer_idx))
        chosen = rng.choice(outer_idx, size=n_pick, replace=False, p=probs)
        mask[chosen] = 1.0
        return torch.from_numpy(mask), num_low_freqs

    mf = create_mask_for_mask_type(
        mask_type, center_fractions=[center_fraction], accelerations=[accel]
    )
    dummy = torch.zeros((1, 1, W, 2), dtype=torch.float32)
    res = T.apply_mask(dummy, mf, seed=seed)
    mask_1d = res[1].reshape(-1).to(torch.float32)
    n_acs = int(res[2]) if len(res) == 3 else num_low_freqs
    return mask_1d, n_acs


def estimate_sens_maps_espirit(
    y_omega: torch.Tensor,
    num_low_freqs: int,
    acs_size: int = 24,
    kernel_size: int = 5,
    eps: float = 1e-8,
) -> torch.Tensor:
    C, H, W = y_omega.shape
    if C == 1:
        return torch.ones_like(y_omega)

    acs_w = min(W, max(8, min(int(num_low_freqs), int(acs_size))))
    acs_h = min(H, max(8, int(acs_size)))
    h0, w0 = (H - acs_h) // 2, (W - acs_w + 1) // 2

    hann_2d = (
        torch.hann_window(acs_h, dtype=torch.float32, device=y_omega.device)[:, None]
        * torch.hann_window(acs_w, dtype=torch.float32, device=y_omega.device)[None, :]
    )
    k_acs = torch.zeros_like(y_omega)
    k_acs[:, h0 : h0 + acs_h, w0 : w0 + acs_w] = (
        y_omega[:, h0 : h0 + acs_h, w0 : w0 + acs_w] * hann_2d
    )

    low_res = ifft2c_complex(k_acs)
    pad = kernel_size // 2
    rss_init = torch.sqrt((low_res.real ** 2 + low_res.imag ** 2).sum(dim=0, keepdim=True) + eps)
    s = low_res / rss_init

    for _ in range(3):
        proj = (s.conj() * low_res).sum(dim=0, keepdim=True)
        proj_re = F.avg_pool2d(proj.real.unsqueeze(0), kernel_size, stride=1, padding=pad).squeeze(0)
        proj_im = F.avg_pool2d(proj.imag.unsqueeze(0), kernel_size, stride=1, padding=pad).squeeze(0)
        proj_s = torch.complex(proj_re, proj_im)
        s_new = low_res * proj_s.conj()
        s_re = F.avg_pool2d(s_new.real.unsqueeze(0), kernel_size, stride=1, padding=pad).squeeze(0)
        s_im = F.avg_pool2d(s_new.imag.unsqueeze(0), kernel_size, stride=1, padding=pad).squeeze(0)
        s_cand = torch.complex(s_re, s_im)
        norm = torch.sqrt((s_cand.real ** 2 + s_cand.imag ** 2).sum(dim=0, keepdim=True) + eps)
        s = s_cand / norm

    ref_phase = torch.exp(-1j * torch.angle(s[0:1] + eps))
    return (s * ref_phase).to(torch.complex64)


# ==============================================================================
# 2. MODELS & CLASSICAL SOLVERS
# ==============================================================================
class MRIPhysics(nn.Module):
    def forward(self, x: torch.Tensor, S: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return mask * fft2c_complex(S * x.unsqueeze(1))

    def adjoint(self, y: torch.Tensor, S: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (S.conj() * ifft2c_complex(mask * y)).sum(dim=1)

    def normal(self, x: torch.Tensor, S: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(x, S, mask), S, mask)


class CGSolver(nn.Module):
    def __init__(self, physics: MRIPhysics, K: int = 6, eps: float = 1e-10):
        super().__init__()
        self.physics = physics
        self.K = int(K)
        self.eps = float(eps)

    @staticmethod
    def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.conj() * b).real.sum(dim=(1, 2), keepdim=True)

    def forward(self, z, AHy, S, mask, mu):
        if self.K <= 0:
            return z
        b = AHy + mu * z
        op = lambda v: self.physics.normal(v, S, mask) + mu * v
        x = z
        r = b - op(x)
        p = r
        rs = self._dot(r, r)
        for _ in range(self.K):
            Ap = op(p)
            alpha = rs / (self._dot(p, Ap) + self.eps)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = self._dot(r, r)
            p = r + (rs_new / (rs + self.eps)) * p
            rs = rs_new
        return x


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.InstanceNorm2d(c_out, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.InstanceNorm2d(c_out, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
    )


class ResidualUNet(nn.Module):
    def __init__(self, in_ch: int = 2, base: int = 32):
        super().__init__()
        c1, c2, c3, cb = base, base * 2, base * 4, base * 8
        self.enc1, self.enc2, self.enc3 = conv_block(in_ch, c1), conv_block(c1, c2), conv_block(c2, c3)
        self.bott = conv_block(c3, cb)
        self.pool = nn.AvgPool2d(2)
        self.up3, self.dec3 = nn.ConvTranspose2d(cb, c3, 2, stride=2), conv_block(2 * c3, c3)
        self.up2, self.dec2 = nn.ConvTranspose2d(c3, c2, 2, stride=2), conv_block(2 * c2, c2)
        self.up1, self.dec1 = nn.ConvTranspose2d(c2, c1, 2, stride=2), conv_block(2 * c1, c1)
        self.out = nn.Conv2d(c1, in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        pad_h, pad_w = (8 - H % 8) % 8, (8 - W % 8) % 8
        x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else x
        e1 = self.enc1(x_in)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return (x_in + self.out(d1))[..., :H, :W]


class MedixUnrolledNet(nn.Module):
    def __init__(
        self,
        T: int = 8,
        K: int = 6,
        base_ch: int = 32,
        mu_init: float = 0.1,
        learn_mu: bool = True,
    ):
        super().__init__()
        self.T = int(T)
        self.physics = MRIPhysics()
        self.denoiser = ResidualUNet(in_ch=2, base=base_ch)
        self.cg = CGSolver(self.physics, K=K)
        init_t = torch.full((self.T,), math.log(mu_init), dtype=torch.float32)
        if learn_mu and K > 0:
            self.log_mu = nn.Parameter(init_t)
        else:
            self.register_buffer("log_mu", init_t)

    def forward(self, y: torch.Tensor, mask: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        AHy = self.physics.adjoint(y, S, mask)
        x = AHy
        for t in range(self.T):
            z = channels_to_complex(self.denoiser(complex_to_channels(x)))
            x = self.cg(z, AHy, S, mask, torch.exp(self.log_mu[t]))
        return x


@torch.no_grad()
def cgsense_espirit_reconstruction(y, mask, S, physics: MRIPhysics, lam: float = 0.01, iters: int = 20):
    cg = CGSolver(physics, K=iters)
    AHy = physics.adjoint(y, S, mask)
    return cg(torch.zeros_like(AHy), AHy, S, mask, lam)


def _tv_prox_chambolle_complex(u: torch.Tensor, weight: float, n_inner: int = 8) -> torch.Tensor:
    if weight <= 0.0:
        return u
    is_3d = u.shape[0] > 1
    tau = 0.125 if is_3d else 0.20
    px = torch.zeros_like(u)
    py = torch.zeros_like(u)
    pz = torch.zeros_like(u) if is_3d else None

    for _ in range(n_inner):
        div = torch.zeros_like(u)
        div[:, :, :-1] += px[:, :, :-1]
        div[:, :, 1:] -= px[:, :, :-1]
        div[:, :-1, :] += py[:, :-1, :]
        div[:, 1:, :] -= py[:, :-1, :]
        if is_3d:
            div[:-1, :, :] += pz[:-1, :, :]
            div[1:, :, :] -= pz[:-1, :, :]

        v = div - u / weight
        gx = torch.zeros_like(u)
        gx[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
        gy = torch.zeros_like(u)
        gy[:, :-1, :] = v[:, 1:, :] - v[:, :-1, :]
        px = px + tau * gx
        py = py + tau * gy
        if is_3d:
            gz = torch.zeros_like(u)
            gz[:-1, :, :] = v[1:, :, :] - v[:-1, :, :]
            pz = pz + tau * gz
            norm = torch.sqrt(
                px.real**2 + px.imag**2 + py.real**2 + py.imag**2 + pz.real**2 + pz.imag**2
            )
            denom = torch.clamp(norm, min=1.0)
            px, py, pz = px / denom, py / denom, pz / denom
        else:
            norm = torch.sqrt(px.real**2 + px.imag**2 + py.real**2 + py.imag**2)
            denom = torch.clamp(norm, min=1.0)
            px, py = px / denom, py / denom

    div = torch.zeros_like(u)
    div[:, :, :-1] += px[:, :, :-1]
    div[:, :, 1:] -= px[:, :, :-1]
    div[:, :-1, :] += py[:, :-1, :]
    div[:, 1:, :] -= py[:, :-1, :]
    if is_3d:
        div[:-1, :, :] += pz[:-1, :, :]
        div[1:, :, :] -= pz[:-1, :, :]
    return u - weight * div


@torch.no_grad()
def tv_cs_reconstruction(y, mask, S, physics: MRIPhysics, lam: float = 0.001, iters: int = 40, eta: float = 1.0):
    AHy = physics.adjoint(y, S, mask)
    x = AHy.clone()
    z = x.clone()
    t = 1.0
    for _ in range(iters):
        grad = physics.normal(z, S, mask) - AHy
        u = z - eta * grad
        x_next = _tv_prox_chambolle_complex(u, weight=eta * lam, n_inner=8)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        z = x_next + ((t - 1.0) / t_next) * (x_next - x)
        x, t = x_next, t_next
    return x


def pad_to_mult(x: torch.Tensor, mult: int = 16):
    H, W = x.shape[-2:]
    ph, pw = (mult - H % mult) % mult, (mult - W % mult) % mult
    if ph or pw:
        return F.pad(x, (0, pw, 0, ph), mode="reflect"), (H, W)
    return x, (H, W)


def fastmri_unet_forward(model, y_ri: torch.Tensor, target_shape) -> torch.Tensor:
    zf_rss = fastmri.rss_complex(fastmri.ifft2c(y_ri), dim=1)
    zf_crop = safe_center_crop_or_pad(zf_rss, target_shape)
    zf_norm, mean, std = T.normalize_instance(zf_crop, eps=1e-11)
    zf_norm = zf_norm.clamp(-6, 6).unsqueeze(1)
    zf_pad, (Ht, Wt) = pad_to_mult(zf_norm, mult=16)
    out = model(zf_pad)[..., :Ht, :Wt].squeeze(1)
    return out * std + mean


def fastmri_varnet_forward(model, y_ri, mask_bool, num_low_freqs, target_shape) -> torch.Tensor:
    sig = inspect.signature(model.forward).parameters
    if "num_low_frequencies" in sig:
        pred_full = model(y_ri, mask_bool, num_low_frequencies=num_low_freqs)
    else:
        pred_full = model(y_ri, mask_bool)
    return safe_center_crop_or_pad(pred_full, target_shape)


# ==============================================================================
# 3. CHECKPOINT LOADER (LOCAL OR HUGGING FACE, ZEROGPU-SAFE CPU CACHE)
# ==============================================================================
def safe_torch_load(path: str, map_location="cpu"):
    sig = inspect.signature(torch.load).parameters
    if "weights_only" in sig:
        return torch.load(path, map_location=map_location, weights_only=False)
    return torch.load(path, map_location=map_location)


def get_deep_model(method_name: str) -> nn.Module:
    if method_name in _MODEL_CACHE:
        return _MODEL_CACHE[method_name]

    filename, arch_type, kwargs = DEEP_METHODS[method_name]
    local_ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", filename)
    ckpt_path = (
        local_ckpt_path
        if os.path.isfile(local_ckpt_path)
        else hf_hub_download(repo_id=MODEL_REPO, filename=filename)
    )
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    if arch_type == "medix":
        model = MedixUnrolledNet(
            T=kwargs["T"], K=kwargs["K"], base_ch=32, mu_init=0.1, learn_mu=(kwargs["K"] > 0)
        )
    elif arch_type == "unet":
        model = FastMRIUnet(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0)
    elif arch_type == "varnet":
        model = FastMRIVarNet(num_cascades=6, sens_chans=8, sens_pools=4, chans=12, pools=4)
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")

    model.load_state_dict(state_dict)
    model.eval()
    _MODEL_CACHE[method_name] = model
    return model


# ==============================================================================
# 4. DATA LOADING (.H5 & .NPZ) & RECONSTRUCTION PIPELINE
# ==============================================================================
def _h5_kspace_layout(shape: Tuple[int, ...], dtype) -> Tuple[Optional[int], int, bool]:
    """Infer (slice_axis, n_slices, ri_last) for arbitrary 2D/3D/4D/5D+ k-space."""
    is_complex = np.issubdtype(dtype, np.complexfloating)
    ri_last = (not is_complex) and (len(shape) >= 3) and (shape[-1] == 2)
    eff_shape = shape[:-1] if ri_last else shape
    ndim = len(eff_shape)

    if ndim < 2:
        raise ValueError(f"Unsupported kspace shape {shape}: expected at least 2 dimensions.")
    if ndim == 2:
        return None, 1, ri_last
    if ndim in (3, 4):
        return 0, int(eff_shape[0]), ri_last

    # 5D+ volumes (e.g., fastMRI Breast: (2, 288, 640, 16, 83))
    if eff_shape[0] <= 10 and eff_shape[1] > eff_shape[0]:
        return 1, int(eff_shape[1]), ri_last
    return 0, int(eff_shape[0]), ri_last


def _read_h5_kspace_slice(ds: h5py.Dataset, slice_idx: int) -> Tuple[np.ndarray, int, int]:
    """Lazily read a single slice from any 2D/3D/4D/5D+ HDF5 kspace dataset as (C, H, W)."""
    slice_axis, n_slices, ri_last = _h5_kspace_layout(ds.shape, ds.dtype)
    s_idx = max(0, min(int(slice_idx), n_slices - 1))
    eff_ndim = len(ds.shape) - (1 if ri_last else 0)

    if eff_ndim == 2:
        slicer = [slice(None), slice(None)]
    elif eff_ndim == 3:
        slicer = [s_idx, slice(None), slice(None)]
    elif eff_ndim == 4:
        slicer = [s_idx, slice(None), slice(None), slice(None)]
    else:
        lead = eff_ndim - 3
        slicer = [0] * lead + [slice(None)] * 3
        slicer[slice_axis] = s_idx

    if ri_last:
        slicer.append(slice(None))
        raw = np.asarray(ds[tuple(slicer)], dtype=np.float32)
        k_np = (raw[..., 0] + 1j * raw[..., 1]).astype(np.complex64)
    else:
        k_np = np.asarray(ds[tuple(slicer)], dtype=np.complex64)

    if k_np.ndim == 2:
        k_np = k_np[None, ...]
    elif k_np.ndim == 3 and (eff_ndim >= 5 or (k_np.shape[0] > 64 and min(k_np.shape) <= 64)):
        # Move the coil axis (smallest dimension) to axis 0 -> (C, H, W)
        coil_axis = int(np.argmin(k_np.shape))
        if coil_axis != 0:
            k_np = np.moveaxis(k_np, coil_axis, 0)

    return k_np, s_idx, n_slices


def _read_h5_gt_slice(ds: h5py.Dataset, s_idx: int) -> Optional[np.ndarray]:
    """Lazily read a matching 2D ground-truth slice from any 2D/3D/4D+ reconstruction dataset."""
    if ds.ndim == 2:
        gt = ds[:]
    elif ds.ndim == 3:
        gt = ds[max(0, min(s_idx, ds.shape[0] - 1))]
    elif ds.ndim >= 4:
        slice_axis = 1 if (ds.shape[0] <= 10 and ds.shape[1] > ds.shape[0]) else 0
        lead = ds.ndim - 2
        slicer = [0] * lead + [slice(None), slice(None)]
        slicer[slice_axis] = max(0, min(s_idx, ds.shape[slice_axis] - 1))
        gt = ds[tuple(slicer)]
    else:
        return None

    gt = np.asarray(gt)
    if np.iscomplexobj(gt):
        gt = np.abs(gt)
    gt = gt.astype(np.float32, copy=False)
    return gt if gt.ndim == 2 else None


def inspect_uploaded_file(fpath: str) -> Tuple[int, str]:
    ext = os.path.splitext(fpath)[1].lower()
    if ext == ".h5":
        with h5py.File(fpath, "r") as hf:
            if "kspace" not in hf:
                raise ValueError("The uploaded .h5 file does not contain a 'kspace' dataset.")
            ds = hf["kspace"]
            _, n_slices, _ = _h5_kspace_layout(ds.shape, ds.dtype)
            acq = str(hf.attrs.get("acquisition", "UNKNOWN"))
            shape_str = str(tuple(ds.shape))
            return n_slices, f"NYU fastMRI HDF5 | Shape: {shape_str} | Contrast: {acq}"
    elif ext == ".npz":
        data = np.load(fpath)
        shape_str = str(tuple(data["y_complex"].shape)) if "y_complex" in data else "N/A"
        return 1, f"Preprocessed Demo NPZ | y_complex: {shape_str}"
    else:
        raise ValueError("Unsupported file format. Please upload a .h5 (NYU fastMRI) or .npz file.")


def load_slice_bundle(
    fpath: str,
    slice_idx: int,
    accel: int,
    mask_type: str,
    acs_size: int,
    device: torch.device,
    seed: int = 42,
) -> dict:
    ext = os.path.splitext(fpath)[1].lower()

    if ext == ".npz":
        data = np.load(fpath)
        y_c = torch.from_numpy(data["y_complex"].astype(np.complex64))
        mask_1d = torch.from_numpy(data["mask_1d"].astype(np.float32)).view(-1)
        sens = torch.from_numpy(data["sens"].astype(np.complex64))
        gt_np = data["target_rss"].astype(np.float32) if "target_rss" in data else None
        scale = float(data["scale"])
        n_acs = int(data["num_low_freqs"])
        y_ri = torch.view_as_real(y_c.contiguous())
        target_shape = gt_np.shape if gt_np is not None else (y_c.shape[-2], y_c.shape[-1])
        info = f"Loaded NPZ (pre-masked) | Coils: {y_c.shape[0]} | Matrix: {y_c.shape[-2]}x{y_c.shape[-1]} | ACS: {n_acs}"
    elif ext == ".h5":
        with h5py.File(fpath, "r") as hf:
            if "kspace" not in hf:
                raise ValueError("The uploaded .h5 file does not contain a 'kspace' dataset.")
            k_np, s_idx, _ = _read_h5_kspace_slice(hf["kspace"], slice_idx)
            t_key = "reconstruction_rss" if "reconstruction_rss" in hf else (
                "reconstruction_esc" if "reconstruction_esc" in hf else None
            )
            gt_np = _read_h5_gt_slice(hf[t_key], s_idx) if t_key else None
            acq = str(hf.attrs.get("acquisition", "UNKNOWN"))

        target_h = max(320, int(gt_np.shape[-2])) if gt_np is not None else 320
        k_np = remove_readout_oversampling_np(k_np, target_h=target_h)
        k_np, energy = svd_coil_compress_volume_np(k_np[None, ...], max_coils=12)
        k_np = k_np[0]

        center_frac = 0.08 if accel <= 4 else 0.04
        mask_1d, n_acs = generate_cartesian_mask_1d(
            W=k_np.shape[-1],
            accel=int(accel),
            center_fraction=center_frac,
            mask_type=mask_type,
            seed=int(seed),
        )
        k_ri = T.to_tensor(k_np.astype(np.complex64))
        masked_ri = k_ri * mask_1d.view(1, 1, -1, 1)
        zf_raw = fastmri.rss_complex(fastmri.ifft2c(masked_ri), dim=0)
        scale = float(zf_raw.max().item()) + 1e-12
        y_ri = masked_ri / scale
        y_c = torch.view_as_complex(y_ri.contiguous())
        sens = estimate_sens_maps_espirit(y_c, n_acs, acs_size=int(acs_size))
        target_shape = gt_np.shape if gt_np is not None else (y_c.shape[-2], y_c.shape[-1])
        info = (
            f"Slice {s_idx} | Contrast: {acq} | Coils: {y_c.shape[0]} (SVD {energy*100:.1f}%) | "
            f"Matrix: {y_c.shape[-2]}x{y_c.shape[-1]} | R={accel}x {mask_type} | ACS={acs_size}x{acs_size}"
        )
    else:
        raise ValueError("Unsupported file format. Please upload a .h5 or .npz file.")

    W = y_c.shape[-1]
    supp_mask = ((sens.real ** 2 + sens.imag ** 2).sum(dim=0) > 0.5).float()

    return {
        "y": y_c.unsqueeze(0).to(device),
        "y_ri": y_ri.unsqueeze(0).to(device),
        "mask": mask_1d.view(1, 1, 1, W).to(device),
        "mask_bool": (mask_1d > 0.5).view(1, 1, 1, W, 1).to(device),
        "num_low_freqs": torch.tensor([n_acs], dtype=torch.long, device=device),
        "sens": sens.unsqueeze(0).to(device),
        "supp_mask": supp_mask.unsqueeze(0).to(device),
        "gt": gt_np,
        "target_shape": target_shape,
        "scale": scale,
        "mask_1d_np": mask_1d.numpy(),
        "info": info,
    }


@torch.no_grad()
def run_single_method(
    method_name: str,
    bundle: dict,
    physics: MRIPhysics,
    device: torch.device,
    cg_steps_medix: int = 6,
    cgsense_lam: float = 0.01,
    cgsense_iters: int = 20,
    tv_lam: float = 0.001,
    tv_iters: int = 40,
    apply_support_mask: bool = False,
) -> Tuple[np.ndarray, float]:
    t_shape = bundle["target_shape"]
    scale = bundle["scale"]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    if method_name == "Zero-Filled (ZF)":
        rec = fastmri.rss_complex(fastmri.ifft2c(bundle["y_ri"]), dim=1)
        out = safe_center_crop_or_pad(rec, t_shape)[0] * scale

    elif method_name == "CG-SENSE (ESPIRiT)":
        rec = cgsense_espirit_reconstruction(
            bundle["y"], bundle["mask"], bundle["sens"], physics, lam=cgsense_lam, iters=int(cgsense_iters)
        ).abs()
        out = safe_center_crop_or_pad(rec, t_shape)[0] * scale

    elif method_name == "TV-CS (FISTA)":
        rec = tv_cs_reconstruction(
            bundle["y"], bundle["mask"], bundle["sens"], physics, lam=tv_lam, iters=int(tv_iters)
        ).abs()
        out = safe_center_crop_or_pad(rec, t_shape)[0] * scale

    else:
        model = get_deep_model(method_name)
        _, arch_type, kwargs = DEEP_METHODS[method_name]
        model.to(device)
        try:
            if arch_type == "medix":
                orig_k = model.cg.K
                if kwargs["K"] > 0:
                    model.cg.K = int(cg_steps_medix)
                rec = model(bundle["y"], bundle["mask"], bundle["sens"]).abs()
                model.cg.K = orig_k
                if apply_support_mask and "SSDU" in method_name:
                    rec = rec * bundle["supp_mask"]
                out = safe_center_crop_or_pad(rec, t_shape)[0] * scale
            elif arch_type == "unet":
                out = fastmri_unet_forward(model, bundle["y_ri"], t_shape)[0] * scale
            elif arch_type == "varnet":
                out = fastmri_varnet_forward(
                    model, bundle["y_ri"], bundle["mask_bool"], bundle["num_low_freqs"], t_shape
                )[0] * scale
            else:
                raise ValueError(f"Unsupported method: {method_name}")
        finally:
            model.to("cpu")

    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return out.detach().cpu().numpy().astype(np.float32), dt_ms


# ==============================================================================
# 5. METRICS, EXPORT & VISUALIZATION TOOLS
# ==============================================================================
def _safe_2d_ssim(gt_2d: np.ndarray, pred_2d: np.ndarray, maxval: float) -> float:
    min_dim = min(gt_2d.shape[0], gt_2d.shape[1])
    if min_dim >= 7:
        try:
            return float(fm_eval.ssim(gt_2d[None, ...], pred_2d[None, ...], maxval=maxval))
        except Exception:
            return float(sk_ssim(gt_2d, pred_2d, data_range=maxval, win_size=7))
    win = max(3, min_dim if (min_dim % 2 == 1) else (min_dim - 1))
    return float(sk_ssim(gt_2d, pred_2d, data_range=maxval, win_size=win))


def eval_slice_metrics(gt_2d: Optional[np.ndarray], pred_2d: np.ndarray) -> dict:
    if gt_2d is None:
        return {"PSNR": None, "SSIM": None, "NMSE": None, "NRMSE": None, "HFEN": None}
    gt_2d = np.asarray(gt_2d, dtype=np.float32)
    pred_2d = np.asarray(pred_2d, dtype=np.float32)
    maxval = float(gt_2d.max()) + 1e-12
    p = float(fm_eval.psnr(gt_2d, pred_2d, maxval=maxval))
    s = _safe_2d_ssim(gt_2d, pred_2d, maxval=maxval)
    n = float(fm_eval.nmse(gt_2d, pred_2d))
    nrmse = float(math.sqrt(max(n, 0.0)))
    log_gt = gaussian_laplace(gt_2d, sigma=1.5)
    log_pr = gaussian_laplace(pred_2d, sigma=1.5)
    hfen = float(np.linalg.norm(log_pr - log_gt) / (np.linalg.norm(log_gt) + 1e-12))
    return {"PSNR": p, "SSIM": s, "NMSE": n, "NRMSE": nrmse, "HFEN": hfen}


def to_display_uint8(img: np.ndarray, vmax: float) -> np.ndarray:
    norm = np.clip(img / (vmax + 1e-12), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def reconstruction_export_files(
    recons: Dict[str, np.ndarray],
    vmax_img: float,
) -> Dict[str, bytes]:
    """Return display PNGs and full-precision float32 NPY files for reconstructed methods."""
    if not np.isfinite(vmax_img) or vmax_img <= 0:
        raise ValueError(f"Image display maximum must be finite and positive, got {vmax_img}.")

    export_files = {}
    for name, image in recons.items():
        if name == "Ground Truth (RSS)":
            continue
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._-").lower() or "reconstruction"
        image_f32 = np.asarray(image, dtype=np.float32)
        if image_f32.ndim != 2:
            raise ValueError(f"Expected a 2D reconstruction for export, got {image_f32.shape}.")
        image_f32 = np.nan_to_num(image_f32, copy=True, nan=0.0, posinf=vmax_img, neginf=0.0)

        png_buffer = io.BytesIO()
        Image.fromarray(to_display_uint8(image_f32, vmax_img)).save(png_buffer, format="PNG", optimize=True)
        npy_buffer = io.BytesIO()
        np.save(npy_buffer, image_f32, allow_pickle=False)
        export_files[f"reconstructions/{slug}.png"] = png_buffer.getvalue()
        export_files[f"reconstructions/{slug}.npy"] = npy_buffer.getvalue()
    return export_files


def plot_metrics_chart(df):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=130)
    has_gt = df["PSNR (dB) ↑"].notna().any()

    if has_gt:
        names = df["Method"].tolist()
        psnrs = df["PSNR (dB) ↑"].tolist()
        hfens = df["HFEN ↓"].tolist()
        lats = df["Latency (ms) ↓"].tolist()

        y_pos = np.arange(len(names))
        axes[0].barh(y_pos, psnrs, color="#0072B2", alpha=0.88)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(names, fontsize=8.5)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("PSNR (dB) ↑")
        axes[0].set_title("Reconstruction Quality (PSNR)")
        axes[0].grid(axis="x", alpha=0.3)

        axes[1].scatter(lats, hfens, s=90, color="#D55E00", edgecolors="black", zorder=4)
        for x_l, y_h, lbl in zip(lats, hfens, names):
            axes[1].annotate(lbl, (x_l, y_h), xytext=(5, 3), textcoords="offset points", fontsize=7.5)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Latency (ms, log scale) ↓")
        axes[1].set_ylabel("HFEN (Edge Error) ↓")
        axes[1].set_title("Edge Sharpness (HFEN) vs. Speed")
        axes[1].grid(True, which="both", alpha=0.25)
    else:
        names = df["Method"].tolist()
        lats = df["Latency (ms) ↓"].tolist()
        y_pos = np.arange(len(names))
        axes[0].barh(y_pos, lats, color="#009E73", alpha=0.88)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(names, fontsize=8.5)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Latency (ms) ↓")
        axes[0].set_title("Inference Latency Comparison")
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No Ground Truth in file\n(PSNR / SSIM / HFEN unavailable)", ha="center", va="center")

    plt.tight_layout()
    return fig


def plot_unified_error_grid(
    recons: Dict[str, np.ndarray],
    gt: Optional[np.ndarray],
    vmax_img: float,
    err_gain: float = 10.0,
):
    method_keys = [k for k in recons.keys() if k != "Ground Truth (RSS)"]
    panels = ([("Ground Truth (RSS)", gt)] if gt is not None else []) + [(k, recons[k]) for k in method_keys]
    ncols = len(panels)

    if gt is None:
        fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.6), dpi=130)
        if ncols == 1:
            axes = [axes]
        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img, cmap="gray", vmin=0, vmax=vmax_img)
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.axis("off")
        plt.tight_layout()
        return fig

    errs = {m: np.abs(recons[m] - gt) * err_gain for m in method_keys}
    vmax_err = float(np.median([np.percentile(e, 99.2) for e in errs.values()])) * 1.35 + 1e-12

    fig = plt.figure(figsize=(3.1 * ncols + 0.6, 6.4), dpi=130)
    gs = fig.add_gridspec(2, ncols + 1, width_ratios=[1.0] * ncols + [0.06], wspace=0.06, hspace=0.14)

    for col, (title, img) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax_img)
        if title != "Ground Truth (RSS)":
            m_res = eval_slice_metrics(gt, img)
            ax.set_title(f"{title}\n{m_res['PSNR']:.2f} dB | SSIM {m_res['SSIM']:.3f}", fontsize=8.2, fontweight="bold")
        else:
            ax.set_title("Ground Truth (RSS)", fontsize=8.5, fontweight="bold")
        ax.axis("off")

    ax_lbl = fig.add_subplot(gs[1, 0])
    ax_lbl.axis("off")
    ax_lbl.text(
        0.5, 0.5, f"Absolute Error\nMaps (x{int(err_gain)})\nUnified Scale",
        ha="center", va="center", fontsize=9.5, fontweight="bold",
    )

    im = None
    for col, m in enumerate(method_keys, start=1):
        ax_e = fig.add_subplot(gs[1, col])
        im = ax_e.imshow(errs[m], cmap="inferno", vmin=0, vmax=vmax_err)
        ax_e.axis("off")

    cbar_ax = fig.add_subplot(gs[1, ncols])
    if im is not None:
        fig.colorbar(im, cax=cbar_ax).set_label(f"{int(err_gain)}x |x_hat - x|", fontsize=8.5)
    return fig


def plot_roi_comparison(
    recons: Dict[str, np.ndarray],
    vmax_img: float,
    x_pct: int,
    y_pct: int,
    box_size: int,
):
    keys = list(recons.keys())
    ncols = len(keys)
    sample_img = recons[keys[0]]
    H, W = sample_img.shape
    sz = max(20, min(int(box_size), H, W))
    c0 = int(np.clip((x_pct / 100.0) * (W - sz), 0, W - sz))
    r0 = int(np.clip((y_pct / 100.0) * (H - sz), 0, H - sz))

    fig, axes = plt.subplots(2, ncols, figsize=(3.1 * ncols, 6.0), dpi=130)
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, name in enumerate(keys):
        img = recons[name]
        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=vmax_img)
        axes[0, col].add_patch(plt.Rectangle((c0, r0), sz, sz, fill=False, edgecolor="#FF3B30", lw=1.6))
        axes[0, col].set_title(name, fontsize=8.5, fontweight="bold")
        axes[0, col].axis("off")

        axes[1, col].imshow(img[r0 : r0 + sz, c0 : c0 + sz], cmap="gray", vmin=0, vmax=vmax_img)
        axes[1, col].set_title(f"Zoomed ROI ({sz}x{sz})", fontsize=8)
        for sp in axes[1, col].spines.values():
            sp.set_edgecolor("#FF3B30")
            sp.set_linewidth(1.5)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

    plt.tight_layout()
    return fig


def plot_ab_difference(img_a: np.ndarray, img_b: np.ndarray, name_a: str, name_b: str, err_gain: float = 10.0):
    diff = np.abs(img_a - img_b) * err_gain
    vmax_d = float(np.percentile(diff, 99.5)) + 1e-12
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.6), dpi=130)
    im = ax.imshow(diff, cmap="inferno", vmin=0, vmax=vmax_d)
    ax.set_title(f"Difference Map: {int(err_gain)}x | {name_a} - {name_b} |", fontsize=9.5, fontweight="bold")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def plot_physics_diagnostics(bundle: dict):
    y_c = bundle["y"][0].detach().cpu()
    sens = bundle["sens"][0].detach().cpu()
    mask_1d = bundle["mask_1d_np"]
    sos = (sens.real ** 2 + sens.imag ** 2).sum(0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.6), dpi=130)

    k_log = np.log1p(y_c[0].abs().numpy() * 100.0)
    axes[0].imshow(k_log, cmap="magma")
    axes[0].set_title("Undersampled k-Space log(1+|y|)", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(mask_1d[None, :], aspect="auto", cmap="Blues", interpolation="nearest")
    axes[1].set_title(f"1D Mask ({int(mask_1d.sum())}/{len(mask_1d)} lines)", fontsize=9)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Phase-Encode Index (ky)")

    im_s = axes[2].imshow(sens[0].abs().numpy(), cmap="viridis")
    axes[2].set_title("ESPIRiT Dominant Coil |S1|", fontsize=9)
    axes[2].axis("off")
    fig.colorbar(im_s, ax=axes[2], fraction=0.046, pad=0.04)

    im_sos = axes[3].imshow(sos, cmap="cividis", vmin=0.0, vmax=1.05)
    axes[3].set_title("Coil Support sum(|S|^2)", fontsize=9)
    axes[3].axis("off")
    fig.colorbar(im_sos, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


# ==============================================================================
# 6. SYNTHETIC 12-COIL PHANTOM GENERATOR & LINEAR-ALGEBRA DIAGNOSTICS
# ==============================================================================
def create_synthetic_phantom_h5(out_path: str = "/tmp/medix_synthetic_12coil_phantom.h5") -> str:
    """Generates a compliant synthetic 12-coil complex brain-like phantom saved as .h5."""
    H, W, C = 320, 320, 12
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, H), np.linspace(-1.0, 1.0, W), indexing="ij")

    img = np.zeros((H, W), dtype=np.float32)
    ellipses = [
        (1.00, 0.69, 0.92, 0.00, 0.00, 0.0),
        (-0.80, 0.6624, 0.874, 0.00, -0.0184, 0.0),
        (-0.20, 0.11, 0.31, 0.22, 0.00, -18.0),
        (-0.20, 0.16, 0.41, -0.22, 0.00, 18.0),
        (0.15, 0.21, 0.25, 0.00, 0.35, 0.0),
        (0.15, 0.046, 0.046, 0.00, 0.10, 0.0),
        (0.15, 0.046, 0.046, 0.00, -0.10, 0.0),
        (0.12, 0.046, 0.023, -0.08, -0.605, 0.0),
        (0.12, 0.023, 0.023, 0.00, -0.606, 0.0),
        (0.12, 0.023, 0.046, 0.06, -0.605, 0.0),
    ]
    for amp, a, b, x0, y0, deg in ellipses:
        th = np.deg2rad(deg)
        xr = (xx - x0) * np.cos(th) + (yy - y0) * np.sin(th)
        yr = -(xx - x0) * np.sin(th) + (yy - y0) * np.cos(th)
        mask_e = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        img[mask_e] += amp

    img = np.clip(img, 0.0, None)
    img = img / (img.max() + 1e-12)

    phase = 0.6 * np.sin(1.7 * xx) * np.cos(1.3 * yy) + 0.3 * xx * yy
    x_complex = (img * np.exp(1j * phase)).astype(np.complex64)

    angles = np.linspace(0, 2 * np.pi, C, endpoint=False)
    sens = np.zeros((C, H, W), dtype=np.complex64)
    for idx, ang in enumerate(angles):
        cx, cy = 1.15 * np.cos(ang), 1.15 * np.sin(ang)
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mag = np.exp(-0.95 * dist2)
        ph = 0.8 * (xx * np.cos(ang) + yy * np.sin(ang))
        sens[idx] = mag * np.exp(1j * ph)
    rss_s = np.sqrt(np.sum(np.abs(sens) ** 2, axis=0, keepdims=True) + 1e-12)
    sens = (sens / rss_s).astype(np.complex64)

    rng = np.random.RandomState(42)
    coil_imgs = sens * x_complex[None, :, :]
    kspace = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(coil_imgs, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    ).astype(np.complex64)
    noise = (rng.normal(0, 0.002, kspace.shape) + 1j * rng.normal(0, 0.002, kspace.shape)).astype(np.complex64)
    kspace = kspace + noise

    gt_rss = np.sqrt(np.sum(np.abs(coil_imgs) ** 2, axis=0)).astype(np.float32)

    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("kspace", data=kspace[None, ...])
        hf.create_dataset("reconstruction_rss", data=gt_rss[None, ...])
        hf.attrs["acquisition"] = "SYNTHETIC_12COIL_PHANTOM"
    return out_path


@torch.no_grad()
def run_la_diagnostics_live(
    model_name: str,
    bundle: dict,
    device: torch.device,
    cg_steps: int = 6,
    n_power_iters: int = 20,
):
    """Computes adjointness, spectral norm bound, per-stage mu_t/kappa_t, and CG Krylov decay."""
    model = get_deep_model(model_name)
    model.to(device)
    try:
        y, mask, S = bundle["y"], bundle["mask"], bundle["sens"]
        gt = bundle["gt"]
        t_shape = bundle["target_shape"]
        scale = bundle["scale"]
        y_norm = torch.linalg.norm(y).item() + 1e-12
        physics = model.physics

        x_rand = torch.randn_like(physics.adjoint(y, S, mask))
        u_rand = torch.randn_like(y) * mask
        lhs = torch.vdot(physics(x_rand, S, mask).reshape(-1), u_rand.reshape(-1))
        rhs = torch.vdot(x_rand.reshape(-1), physics.adjoint(u_rand, S, mask).reshape(-1))
        adj_err = float((lhs - rhs).abs().item() / (lhs.abs().item() + 1e-12))

        v = torch.randn_like(x_rand)
        v = v / (torch.linalg.norm(v) + 1e-12)
        lam_max = 0.0
        for _ in range(n_power_iters):
            Av = physics.normal(v, S, mask)
            lam_max = float(torch.vdot(v.reshape(-1), Av.reshape(-1)).real.item())
            v = Av / (torch.linalg.norm(Av) + 1e-12)

        AHy = physics.adjoint(y, S, mask)
        x_cur = AHy
        K_eff = max(1, int(cg_steps))
        rows = []
        cg_curves = []

        p0 = (safe_center_crop_or_pad(x_cur.abs(), t_shape)[0] * scale).cpu().numpy()
        m0 = eval_slice_metrics(gt, p0)
        dc0 = float(torch.linalg.norm(physics(x_cur, S, mask) - y).item() / y_norm)
        rows.append({
            "Stage t": 0, "Learned mu_t": None, "Kappa Bound": None,
            "CG Rel. Res.": 1.0, "Slice PSNR (dB)": round(m0["PSNR"], 2) if m0["PSNR"] else None,
            "Rel. k-Space DC": round(dc0, 5),
        })

        for t in range(model.T):
            mu_t = torch.exp(model.log_mu[t])
            mu_val = float(mu_t.item())
            kappa_t = (1.0 + mu_val) / mu_val
            z_t = channels_to_complex(model.denoiser(complex_to_channels(x_cur)))

            b_rhs = AHy + mu_t * z_t
            op = lambda vec: physics.normal(vec, S, mask) + mu_t * vec
            x_cg = z_t
            r = b_rhs - op(x_cg)
            p = r
            rs = (r.conj() * r).real.sum(dim=(1, 2), keepdim=True)
            r0_norm = torch.sqrt(rs.clamp_min(0.0)).mean().item() + 1e-12
            hist = [1.0]
            for _ in range(K_eff):
                Ap = op(p)
                alpha = rs / ((p.conj() * Ap).real.sum(dim=(1, 2), keepdim=True) + 1e-10)
                x_cg = x_cg + alpha * p
                r = r - alpha * Ap
                rs_new = (r.conj() * r).real.sum(dim=(1, 2), keepdim=True)
                p = r + (rs_new / (rs + 1e-10)) * p
                rs = rs_new
                hist.append(float(torch.sqrt(rs.clamp_min(0.0)).mean().item() / r0_norm))

            x_cur = x_cg
            cg_curves.append(hist)
            pt = (safe_center_crop_or_pad(x_cur.abs(), t_shape)[0] * scale).cpu().numpy()
            mt = eval_slice_metrics(gt, pt)
            dc_t = float(torch.linalg.norm(physics(x_cur, S, mask) - y).item() / y_norm)
            rows.append({
                "Stage t": t + 1,
                "Learned mu_t": round(mu_val, 4),
                "Kappa Bound": round(kappa_t, 2),
                "CG Rel. Res.": float(f"{hist[-1]:.3e}"),
                "Slice PSNR (dB)": round(mt["PSNR"], 2) if mt["PSNR"] else None,
                "Rel. k-Space DC": round(dc_t, 5),
            })
    finally:
        model.to("cpu")

    stages = [r["Stage t"] for r in rows if r["Stage t"] > 0]
    mus = [r["Learned mu_t"] for r in rows if r["Stage t"] > 0]
    kappas = [r["Kappa Bound"] for r in rows if r["Stage t"] > 0]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), dpi=130)

    l1 = axes[0].plot(stages, mus, "o-", color="#0072B2", lw=2.0, label="Learned mu_t")
    axes[0].set_xlabel("Unrolling Stage t")
    axes[0].set_ylabel("Learned mu_t", color="#0072B2")
    axes[0].set_xticks(stages)
    axes[0].grid(alpha=0.3)
    ax0_r = axes[0].twinx()
    l2 = ax0_r.plot(stages, kappas, "s--", color="#D55E00", lw=1.8, label="Kappa <= (1+mu_t)/mu_t")
    ax0_r.set_ylabel("HPD Condition Bound kappa_t", color="#D55E00")
    axes[0].legend(l1 + l2, [x.get_label() for x in (l1 + l2)], loc="best", fontsize=7.5)
    axes[0].set_title(f"(a) {model_name}: mu_t & kappa_t", fontsize=9.5)

    ks = np.arange(len(cg_curves[0]))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.90, len(cg_curves)))
    for idx_t, curve in enumerate(cg_curves):
        lbl = f"Stage t={idx_t+1}" if idx_t in (0, len(cg_curves) // 2, len(cg_curves) - 1) else None
        axes[1].semilogy(ks, curve, "-o", markersize=3.5, color=cmap[idx_t], lw=1.4, label=lbl)
    rho = (math.sqrt(kappas[-1]) - 1.0) / (math.sqrt(kappas[-1]) + 1.0)
    theory = [min(1.0, 2.0 * (rho ** k)) for k in ks]
    axes[1].semilogy(ks, theory, "--", color="#D55E00", lw=2.0, label="Krylov Bound")
    axes[1].set_xlabel("CG Iteration k")
    axes[1].set_ylabel("||r_k||_2 / ||r_0||_2")
    axes[1].set_title("(b) Inner CG Residual Decay", fontsize=9.5)
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=7.5)

    all_t = [r["Stage t"] for r in rows]
    dcs = [r["Rel. k-Space DC"] for r in rows]
    psnrs = [r["Slice PSNR (dB)"] for r in rows]
    if all(p is not None for p in psnrs):
        axes[2].plot(all_t, psnrs, "o-", color="#009E73", lw=2.0, label="Slice PSNR (dB)")
        axes[2].set_ylabel("Slice PSNR (dB)", color="#009E73")
        ax2_r = axes[2].twinx()
        ax2_r.plot(all_t, dcs, "^--", color="#CC79A7", lw=1.8, label="Rel. k-Space DC")
        ax2_r.set_ylabel("Rel. k-Space DC Residual", color="#CC79A7")
    else:
        axes[2].plot(all_t, dcs, "^--", color="#CC79A7", lw=1.8, label="Rel. k-Space DC")
        axes[2].set_ylabel("Rel. k-Space DC Residual")
    axes[2].set_xlabel("Unrolling Stage t")
    axes[2].set_xticks(all_t)
    axes[2].set_title("(c) Cascade Trajectory", fontsize=9.5)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    summary = (
        f"Adjoint Error: {adj_err:.2e} (float32 exact) | "
        f"Spectral Norm ||A||_2^2: {lam_max:.6f} (Theory <= 1.0) | "
        f"Model: {model_name}"
    )
    return fig, rows, summary

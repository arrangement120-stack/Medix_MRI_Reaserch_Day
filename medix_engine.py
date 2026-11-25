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

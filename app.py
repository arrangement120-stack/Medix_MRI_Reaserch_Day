import gc
import inspect
import io
import json
import os
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import gradio as gr

from medix_engine import (
    ALL_METHODS,
    CLASSICAL_METHODS,
    MODEL_REPO,
    MRIPhysics,
    create_synthetic_phantom_h5,
    eval_slice_metrics,
    get_device,
    inspect_uploaded_file,
    load_slice_bundle,
    plot_ab_difference,
    plot_metrics_chart,
    plot_physics_diagnostics,
    plot_roi_comparison,
    plot_unified_error_grid,
    reconstruction_export_files,
    run_la_diagnostics_live,
    run_single_method,
    to_display_uint8,
)

PHYSICS = MRIPhysics()

# --- Unrestricted Runtime Config for Google Colab, Kaggle, and Local Docker ---
IS_COLAB = "COLAB_RELEASE_TAG" in os.environ or os.path.isdir("/content")
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.isdir("/kaggle/working")
DEFAULT_SHARE = "true" if (IS_COLAB or IS_KAGGLE) else "false"
GRADIO_SHARE = os.getenv("GRADIO_SHARE", DEFAULT_SHARE).lower() in {"1", "true", "yes"}
MAX_CONCURRENT_JOBS = 1
MAX_QUEUE_SIZE = 10
EXPORT_DIR = "/tmp/medix_exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

CORE_BENCHMARK = [
    "Zero-Filled (ZF)",
    "CG-SENSE (ESPIRiT)",
    "Medix (SSDU - T=8)",
    "Unrolled-CG / MoDL (Sup UB)",
]

LA_MODELS = [
    "Medix (SSDU - T=8)",
    "Unrolled-CG / MoDL (Sup UB)",
    "Medix (SSDU - T=6)",
    "Medix (SSDU - T=4)",
]

CUSTOM_THEME = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#f1f5f9",
    block_background_fill="#ffffff",
    block_border_color="#e2e8f0",
    block_border_width="1px",
    block_radius="14px",
    block_shadow="0 4px 14px -4px rgba(15, 23, 42, 0.08)",
    button_primary_background_fill="linear-gradient(135deg, #0284c7 0%, #06b6d4 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #0369a1 0%, #0891b2 100%)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#f8fafc",
    button_secondary_border_color="#cbd5e1",
    button_secondary_text_color="#0f172a",
)

CUSTOM_CSS = """
.gradio-container {
    max-width: 1460px !important;
    margin: 0 auto !important;
}
.medix-hero,
.medix-hero * {
    color: #ffffff !important;
}
.medix-hero {
    background: linear-gradient(135deg, #0b192c 0%, #0e4b6e 55%, #0891b2 100%) !important;
    border-radius: 18px !important;
    padding: 26px 30px !important;
    box-shadow: 0 12px 30px -10px rgba(12, 74, 110, 0.45) !important;
    margin-bottom: 14px !important;
    border: 1px solid rgba(125, 211, 252, 0.45) !important;
}
.medix-hero h1 {
    margin: 0 0 8px 0 !important;
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    color: #ffffff !important;
    text-shadow: 0 2px 6px rgba(0, 0, 0, 0.35) !important;
}
.medix-hero p {
    margin: 0 0 14px 0 !important;
    color: #f1f5f9 !important;
    font-size: 0.96rem !important;
    line-height: 1.55 !important;
    max-width: 940px !important;
}
.medix-hero code {
    background: rgba(255, 255, 255, 0.16) !important;
    color: #ffffff !important;
    padding: 2px 6px !important;
    border-radius: 5px !important;
}
.medix-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.medix-badge {
    display: inline-flex;
    align-items: center;
    padding: 5px 12px !important;
    border-radius: 999px !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    background: #0f172a !important;
    border: 1px solid #38bdf8 !important;
    color: #ffffff !important;
}
.medix-badge-accent {
    background: #0284c7 !important;
    border-color: #7dd3fc !important;
    color: #ffffff !important;
}
.section-tag {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.9rem;
    font-weight: 700;
    color: #0f172a;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin: 6px 0 4px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #e0f2fe;
}
.section-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 7px;
    background: linear-gradient(135deg, #0284c7, #06b6d4);
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 700;
}
.file-pill {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    color: #0c4a6e;
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 0.84rem;
    font-weight: 500;
}
.file-pill-error {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 0.84rem;
    font-weight: 500;
}
.status-banner {
    background: #ffffff;
    border-left: 4px solid #06b6d4;
    border-top: 1px solid #e2e8f0;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 11px 16px;
    color: #0f172a;
    font-size: 0.88rem;
    margin-bottom: 8px;
}
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 12px;
}
@media (max-width: 900px) {
    .kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 2px 8px -2px rgba(15, 23, 42, 0.06);
    border-top: 3px solid #0284c7;
}
.kpi-card.teal { border-top-color: #0d9488; }
.kpi-card.indigo { border-top-color: #4f46e5; }
.kpi-card.amber { border-top-color: #d97706; }
.kpi-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 1.35rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.2;
}
.kpi-sub {
    font-size: 0.76rem;
    color: #475569;
    margin-top: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
button.primary {
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    min-height: 46px !important;
    box-shadow: 0 6px 16px -4px rgba(2, 132, 199, 0.45) !important;
}
.tabs > .tab-nav > button.selected {
    color: #0284c7 !important;
    border-bottom: 2px solid #0284c7 !important;
    font-weight: 700 !important;
}
"""


def get_filepath(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name")
    return getattr(file_obj, "name", None)


def format_file_size(size_mb: float) -> str:
    return f"{size_mb / 1024.0:.2f} GB" if size_mb >= 1024.0 else f"{size_mb:.1f} MB"


def validate_file_exists(fpath: str) -> float:
    """Checks that the selected file exists and is non-empty (no artificial size cap)."""
    if not fpath or not os.path.isfile(fpath):
        raise ValueError("File could not be found on disk. Please check the path or upload again.")
    size_mb = os.path.getsize(fpath) / (1024.0 ** 2)
    if size_mb <= 0:
        raise ValueError("Selected file is empty (0 bytes).")
    return size_mb


def resolve_active_file(active_state, file_obj, local_path_str: str = ""):
    """Resolves filepath from state, uploaded file, or direct Disk/Google Drive path."""
    if isinstance(active_state, dict) and active_state.get("path"):
        return active_state["path"]
    fpath = get_filepath(file_obj)
    if fpath:
        return fpath
    if local_path_str and local_path_str.strip():
        return local_path_str.strip().strip('"').strip("'")
    return None


def cleanup_old_exports(max_keep: int = 16):
    """Prevents /tmp/medix_exports from accumulating stale CSV/ZIP files across sessions."""
    try:
        files = sorted(
            (
                os.path.join(EXPORT_DIR, f)
                for f in os.listdir(EXPORT_DIR)
                if f.endswith((".csv", ".zip"))
            ),
            key=os.path.getmtime,
        )
        for old_f in files[:-max_keep]:
            try:
                os.remove(old_f)
            except OSError:
                pass
    except OSError:
        pass


def create_results_zip(
    run_id: str,
    metrics_df: pd.DataFrame,
    settings: dict,
    recons: dict,
    vmax_img: float,
    figures: dict,
) -> str:
    """Packages metrics CSV, run metadata JSON, diagnostic PNG plots, and reconstructed PNG/NPY files."""
    archive_path = os.path.join(EXPORT_DIR, f"medix_results_{run_id}.zip")
    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": settings,
        "contents": [
            "metrics.csv",
            "run_info.json",
            "figures/*.png",
            "reconstructions/*.png",
            "reconstructions/*.npy",
        ],
        "source_scan_included": False,
    }

    with io.BytesIO() as buf:
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metrics.csv", metrics_df.to_csv(index=False).encode("utf-8-sig"))
            zf.writestr("run_info.json", json.dumps(metadata, indent=2).encode("utf-8"))

            for fig_name, fig_obj in figures.items():
                if fig_obj is not None:
                    fig_buf = io.BytesIO()
                    fig_obj.savefig(fig_buf, format="png", dpi=150, bbox_inches="tight")
                    zf.writestr(f"figures/{fig_name}.png", fig_buf.getvalue())

            for rel_path, raw_bytes in reconstruction_export_files(recons, vmax_img).items():
                zf.writestr(rel_path, raw_bytes)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=EXPORT_DIR, prefix=".medix-", suffix=".tmp", delete=False
            ) as tmp_f:
                tmp_path = tmp_f.name
                tmp_f.write(buf.getvalue())
            os.replace(tmp_path, archive_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return archive_path


def release_memory(device=None):
    """Frees Python objects and clears CUDA cache after each run."""
    gc.collect()
    if device is not None and getattr(device, "type", "") == "cuda":
        torch.cuda.empty_cache()


def to_rgb_uint8(img: np.ndarray, vmax: float) -> np.ndarray:
    u8 = to_display_uint8(img, vmax)
    return np.stack([u8, u8, u8], axis=-1)


def build_idle_kpi_html():
    return """
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Best PSNR</div>
            <div class="kpi-value">-- dB</div>
            <div class="kpi-sub">Awaiting run</div>
        </div>
        <div class="kpi-card teal">
            <div class="kpi-label">Best SSIM</div>
            <div class="kpi-value">--</div>
            <div class="kpi-sub">Awaiting run</div>
        </div>
        <div class="kpi-card indigo">
            <div class="kpi-label">Lowest HFEN (Edge)</div>
            <div class="kpi-value">--</div>
            <div class="kpi-sub">Awaiting run</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-label">Fastest Solver</div>
            <div class="kpi-value">-- ms</div>
            <div class="kpi-sub">Awaiting run</div>
        </div>
    </div>
    """


def build_kpi_html(df: pd.DataFrame):
    has_gt = df["PSNR (dB) ↑"].notna().any()
    fast_row = df.loc[df["Latency (ms) ↓"].idxmin()]

    if has_gt:
        best_psnr = df.loc[df["PSNR (dB) ↑"].idxmax()]
        best_ssim = df.loc[df["SSIM ↑"].idxmax()]
        best_hfen = df.loc[df["HFEN ↓"].idxmin()]
        psnr_val, psnr_sub = f"{best_psnr['PSNR (dB) ↑']:.2f} dB", best_psnr["Method"]
        ssim_val, ssim_sub = f"{best_ssim['SSIM ↑']:.4f}", best_ssim["Method"]
        hfen_val, hfen_sub = f"{best_hfen['HFEN ↓']:.4f}", best_hfen["Method"]
    else:
        psnr_val, psnr_sub = "N/A", "No Ground Truth"
        ssim_val, ssim_sub = "N/A", "No Ground Truth"
        hfen_val, hfen_sub = "N/A", "No Ground Truth"

    fast_val, fast_sub = f"{fast_row['Latency (ms) ↓']:.1f} ms", fast_row["Method"]

    return f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Best PSNR ↑</div>
            <div class="kpi-value">{psnr_val}</div>
            <div class="kpi-sub">{psnr_sub}</div>
        </div>
        <div class="kpi-card teal">
            <div class="kpi-label">Best SSIM ↑</div>
            <div class="kpi-value">{ssim_val}</div>
            <div class="kpi-sub">{ssim_sub}</div>
        </div>
        <div class="kpi-card indigo">
            <div class="kpi-label">Lowest HFEN (Edge) ↓</div>
            <div class="kpi-value">{hfen_val}</div>
            <div class="kpi-sub">{hfen_sub}</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-label">Fastest Inference ↓</div>
            <div class="kpi-value">{fast_val}</div>
            <div class="kpi-sub">{fast_sub}</div>
        </div>
    </div>
    """


def on_file_change(file_obj, active_state):
    fpath = get_filepath(file_obj)
    if not fpath:
        if isinstance(active_state, dict) and active_state.get("is_local"):
            return active_state, gr.update(), gr.update()
        return (
            None,
            gr.update(minimum=0, maximum=1, value=0, step=1, visible=False),
            '<div class="file-pill">Upload any <b>.h5</b> / <b>.npz</b> file (no size limit), load directly from <b>Disk / Google Drive</b>, or click <b>Load Synthetic 12-Coil Phantom</b>.</div>',
        )
    try:
        size_mb = validate_file_exists(fpath)
        n_slices, desc = inspect_uploaded_file(fpath)
        size_label = format_file_size(size_mb)
        new_state = {"path": fpath, "is_local": False}
        if n_slices <= 1:
            return (
                new_state,
                gr.update(minimum=0, maximum=1, value=0, step=1, visible=False),
                f'<div class="file-pill"><b>Ready:</b> {desc} (Single Slice, {size_label})</div>',
            )
        mid = n_slices // 2
        return (
            new_state,
            gr.update(minimum=0, maximum=n_slices - 1, value=mid, step=1, visible=True),
            f'<div class="file-pill"><b>Ready:</b> {desc} ({n_slices} Slices, {size_label} — single-slice lazy read)</div>',
        )
    except Exception as e:
        return (
            None,
            gr.update(minimum=0, maximum=1, value=0, step=1, visible=False),
            f'<div class="file-pill-error"><b>File Error:</b> {e}</div>',
        )


def on_load_local_path(local_path_str: str):
    cleaned = (local_path_str or "").strip().strip('"').strip("'")
    if not cleaned:
        return (
            None,
            gr.update(minimum=0, maximum=1, value=0, step=1, visible=False),
            '<div class="file-pill-error"><b>Path Error:</b> Please enter a valid path to a .h5 or .npz file.</div>',
        )
    try:
        size_mb = validate_file_exists(cleaned)
        n_slices, desc = inspect_uploaded_file(cleaned)
        size_label = format_file_size(size_mb)
        new_state = {"path": cleaned, "is_local": True}
        if n_slices <= 1:
            return (
                new_state,
                gr.update(minimum=0, maximum=1, value=0, step=1, visible=False),
                f'<div class="file-pill"><b>Direct Disk Load:</b> {desc} (Single Slice, {size_label})</div>',
            )
        mid = n_slices // 2
        return (
            new_state,
            gr.update(minimum=0, maximum=n_slices - 1, value=mid, step=1, visible=True),
            f'<div class="file-pill"><b>Direct Disk Load:</b> {desc} ({n_slices} Slices, {size_label} — single-slice lazy read)</div>',
        )
    except Exception as e:
        return (
            None,
            gr.update(minimum=0, maximum=1, value=0, step=1, visible=False),
            f'<div class="file-pill-error"><b>Path Error:</b> {e}</div>',
        )


def on_load_phantom():
    return create_synthetic_phantom_h5()


def run_benchmark(
    active_file_state,
    file_obj,
    local_path_str,
    slice_idx,
    selected_methods,
    accel,
    mask_type,
    acs_size,
    cg_steps_medix,
    tv_iters,
    apply_support_mask,
    err_gain,
    roi_x,
    roi_y,
    roi_size,
    progress=gr.Progress(track_tqdm=False),
):
    fpath = resolve_active_file(active_file_state, file_obj, local_path_str)
    if not fpath:
        raise gr.Error("Please upload a file (.h5 or .npz), load a Local/Drive path, or load the Synthetic 12-Coil Phantom first.")
    if not selected_methods:
        raise gr.Error("Please select at least one reconstruction method.")

    try:
        size_mb = validate_file_exists(fpath)
    except ValueError as e:
        raise gr.Error(str(e))

    plt.close("all")
    device = get_device()
    bundle = None
    try:
        progress(0.05, desc="Loading single slice & estimating ESPIRiT coil maps...")
        with torch.inference_mode():
            bundle = load_slice_bundle(
                fpath=fpath,
                slice_idx=int(slice_idx or 0),
                accel=int(accel),
                mask_type=mask_type,
                acs_size=int(acs_size),
                device=device,
            )

            gt = bundle["gt"]
            recons = {}
            if gt is not None:
                recons["Ground Truth (RSS)"] = np.asarray(gt, dtype=np.float32)

            rows = []
            total_m = len(selected_methods)
            for idx_m, m in enumerate(selected_methods, start=1):
                progress(0.12 + 0.65 * (idx_m - 1) / max(total_m, 1), desc=f"Running ({idx_m}/{total_m}): {m}...")
                rec, dt_ms = run_single_method(
                    method_name=m,
                    bundle=bundle,
                    physics=PHYSICS,
                    device=device,
                    cg_steps_medix=int(cg_steps_medix),
                    tv_iters=int(tv_iters),
                    apply_support_mask=bool(apply_support_mask),
                )
                rec_f32 = np.asarray(rec, dtype=np.float32)
                recons[m] = rec_f32
                scores = eval_slice_metrics(gt, rec_f32)
                rows.append({
                    "Method": m,
                    "PSNR (dB) ↑": round(scores["PSNR"], 2) if scores["PSNR"] is not None else None,
                    "SSIM ↑": round(scores["SSIM"], 4) if scores["SSIM"] is not None else None,
                    "NMSE ↓": round(scores["NMSE"], 5) if scores["NMSE"] is not None else None,
                    "NRMSE ↓": round(scores["NRMSE"], 4) if scores["NRMSE"] is not None else None,
                    "HFEN ↓": round(scores["HFEN"], 4) if scores["HFEN"] is not None else None,
                    "Latency (ms) ↓": round(dt_ms, 1),
                })
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        progress(0.80, desc="Rendering diagnostic figures & packaging downloadable results...")
        df = pd.DataFrame(rows)
        if df["PSNR (dB) ↑"].notna().any():
            df = df.sort_values("PSNR (dB) ↑", ascending=False).reset_index(drop=True)

        ref_for_scale = gt if gt is not None else recons[selected_methods[0]]
        vmax_img = float(np.percentile(ref_for_scale, 99.7)) + 1e-12

        state = {
            "recons": recons,
            "gt": np.asarray(gt, dtype=np.float32) if gt is not None else None,
            "vmax_img": vmax_img,
            "err_gain": float(err_gain),
        }

        choices = list(recons.keys())
        left_default = choices[0]
        right_default = choices[1] if len(choices) > 1 else choices[0]

        slider_imgs = (
            to_rgb_uint8(recons[left_default], vmax_img),
            to_rgb_uint8(recons[right_default], vmax_img),
        )
        ab_diff_fig = plot_ab_difference(
            recons[left_default], recons[right_default], left_default, right_default, float(err_gain)
        )
        metrics_fig = plot_metrics_chart(df)
        error_grid_fig = plot_unified_error_grid(recons, gt, vmax_img, float(err_gain))
        roi_fig = plot_roi_comparison(recons, vmax_img, int(roi_x), int(roi_y), int(roi_size))
        phys_fig = plot_physics_diagnostics(bundle)

        cleanup_old_exports()
        run_id = uuid.uuid4().hex[:8]
        csv_path = os.path.join(EXPORT_DIR, f"medix_metrics_{run_id}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        env_tag = "Colab" if IS_COLAB else ("Kaggle" if IS_KAGGLE else "Local/Docker")
        runtime_tag = f"{env_tag} ({device.type.upper()})"
        settings = {
            "runtime": runtime_tag,
            "methods": list(selected_methods),
            "accel": int(accel),
            "mask_type": str(mask_type),
            "acs_size": int(acs_size),
            "cg_steps": int(cg_steps_medix),
            "tv_iterations": int(tv_iters),
            "coil_support_mask": bool(apply_support_mask),
            "error_map_gain": float(err_gain),
            "slice_index": int(slice_idx or 0),
            "input_size_mb": round(size_mb, 2),
        }
        zip_path = create_results_zip(
            run_id=run_id,
            metrics_df=df,
            settings=settings,
            recons=recons,
            vmax_img=vmax_img,
            figures={
                "metrics_chart": metrics_fig,
                "unified_error_grid": error_grid_fig,
                "roi_comparison": roi_fig,
                "physics_diagnostics": phys_fig,
            },
        )

        info_str = bundle["info"]
        status_html = (
            f'<div class="status-banner">'
            f'<b>Completed on {runtime_tag}</b> &nbsp;|&nbsp; '
            f'Input: {format_file_size(size_mb)} &nbsp;|&nbsp; {info_str}'
            f'</div>'
        )
        kpi_html = build_kpi_html(df)
        progress(1.0, desc="Done.")

        return (
            state,
            status_html,
            kpi_html,
            df,
            csv_path,
            zip_path,
            metrics_fig,
            gr.update(choices=choices, value=left_default),
            gr.update(choices=choices, value=right_default),
            slider_imgs,
            ab_diff_fig,
            error_grid_fig,
            roi_fig,
            phys_fig,
        )
    except torch.cuda.OutOfMemoryError:
        release_memory(device)
        raise gr.Error(
            "GPU memory limit reached for this combination. Try selecting fewer models at once."
        )
    finally:
        if bundle is not None:
            del bundle
        release_memory(device)


def run_la_tab(
    active_file_state,
    file_obj,
    local_path_str,
    slice_idx,
    accel,
    mask_type,
    acs_size,
    cg_steps_medix,
    la_model_name,
    progress=gr.Progress(track_tqdm=False),
):
    fpath = resolve_active_file(active_file_state, file_obj, local_path_str)
    if not fpath:
        raise gr.Error("Upload a scan, load a Local/Drive path, or click 'Load Synthetic 12-Coil Phantom' first.")
    try:
        validate_file_exists(fpath)
    except ValueError as e:
        raise gr.Error(str(e))

    plt.close("all")
    device = get_device()
    bundle = None
    try:
        progress(0.15, desc="Loading slice & running adjoint/spectral checks...")
        with torch.inference_mode():
            bundle = load_slice_bundle(
                fpath=fpath,
                slice_idx=int(slice_idx or 0),
                accel=int(accel),
                mask_type=mask_type,
                acs_size=int(acs_size),
                device=device,
            )
            progress(0.55, desc=f"Tracing Krylov convergence for {la_model_name}...")
            fig, rows, summary = run_la_diagnostics_live(
                model_name=la_model_name,
                bundle=bundle,
                device=device,
                cg_steps=int(cg_steps_medix),
            )
        la_df = pd.DataFrame(rows)
        cleanup_old_exports()
        la_csv_path = os.path.join(EXPORT_DIR, f"medix_la_diag_{uuid.uuid4().hex[:8]}.csv")
        la_df.to_csv(la_csv_path, index=False, encoding="utf-8-sig")
        progress(1.0, desc="Done.")
        return (
            f'<div class="status-banner"><b>Linear-Algebra Verification:</b> {summary}</div>',
            fig,
            la_df,
            la_csv_path,
        )
    except torch.cuda.OutOfMemoryError:
        release_memory(device)
        raise gr.Error("GPU memory limit reached during diagnostics. Try fewer CG steps.")
    finally:
        if bundle is not None:
            del bundle
        release_memory(device)


def on_ab_change(state, left_name, right_name):
    if not state or left_name not in state["recons"] or right_name not in state["recons"]:
        return None, None
    plt.close("all")
    recons = state["recons"]
    vmax = state["vmax_img"]
    slider_imgs = (
        to_rgb_uint8(recons[left_name], vmax),
        to_rgb_uint8(recons[right_name], vmax),
    )
    diff_fig = plot_ab_difference(
        recons[left_name], recons[right_name], left_name, right_name, state["err_gain"]
    )
    return slider_imgs, diff_fig


def on_roi_change(state, roi_x, roi_y, roi_size):
    if not state:
        return None
    plt.close("all")
    return plot_roi_comparison(state["recons"], state["vmax_img"], int(roi_x), int(roi_y), int(roi_size))


_launch_params = inspect.signature(gr.Blocks.launch).parameters
_blocks_kwargs = {"title": "Medix — Multi-Coil MRI Reconstruction Studio"}
_launch_kwargs = {"share": GRADIO_SHARE}
if "theme" in _launch_params:
    _launch_kwargs["theme"] = CUSTOM_THEME
    _launch_kwargs["css"] = CUSTOM_CSS
else:
    _blocks_kwargs["theme"] = CUSTOM_THEME
    _blocks_kwargs["css"] = CUSTOM_CSS


with gr.Blocks(**_blocks_kwargs) as demo:
    results_state = gr.State(value=None)
    active_file_state = gr.State(value=None)

    gr.HTML(
        f"""
        <div class="medix-hero">
            <h1>Medix: Physics-Guided Multi-Coil MRI Reconstruction Studio</h1>
            <p>
                Unrestricted benchmark workbench for Google Colab, Kaggle, and Local Docker.
                Upload raw NYU fastMRI (<code>.h5</code>) or preprocessed (<code>.npz</code>) volumes of any size,
                load directly from Disk / Google Drive via single-slice lazy reading, or test the built-in synthetic 12-coil phantom.
            </p>
            <div class="medix-badges">
                <span class="medix-badge medix-badge-accent">Colab / Kaggle / Docker Edition</span>
                <span class="medix-badge">No File Size Limit</span>
                <span class="medix-badge">Weights: {MODEL_REPO}</span>
                <span class="medix-badge">API &amp; Downloadable Results (.zip &amp; .csv)</span>
            </div>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=340):
            gr.HTML('<div class="section-tag"><span class="section-num">1</span> Data Input</div>')
            file_in = gr.File(
                label="Upload k-Space File (.h5 or .npz — No Size Limit)",
                file_types=[".h5", ".npz"],
            )
            with gr.Accordion("Load Directly from Disk / Google Drive (Recommended for 1+ GB Volumes)", open=True):
                local_path_in = gr.Textbox(
                    label="Local / Google Drive File Path (.h5 or .npz)",
                    placeholder="/content/drive/MyDrive/fastmri_samples/knee/multicoil_val_file1001566.h5",
                )
                btn_load_local = gr.Button("Load From Path (Lazy Single-Slice Read)", size="sm")
            btn_phantom = gr.Button("Load Synthetic 12-Coil Brain Phantom (No Patient Data)", size="sm")
            file_info = gr.HTML(
                '<div class="file-pill">Upload any <b>.h5</b> / <b>.npz</b> file (no size limit), load directly from <b>Disk / Google Drive</b>, or click <b>Load Synthetic 12-Coil Phantom</b>.</div>'
            )
            slice_slider = gr.Slider(
                minimum=0,
                maximum=1,
                value=0,
                step=1,
                visible=False,
                label="Slice Index (Multi-Slice .h5)",
            )

            gr.HTML('<div class="section-tag"><span class="section-num">2</span> Reconstruction Methods</div>')
            with gr.Row():
                btn_core = gr.Button("Core 4", size="sm")
                btn_classical = gr.Button("Classical", size="sm")
                btn_all = gr.Button("Select All", size="sm")
                btn_clear = gr.Button("Clear", size="sm")

            methods_cb = gr.CheckboxGroup(
                choices=ALL_METHODS,
                value=CORE_BENCHMARK,
                label="Choose One or More Methods to Benchmark",
            )

            with gr.Accordion("3. Physics, Sampling & Ablation Controls", open=False):
                with gr.Row():
                    accel_radio = gr.Radio(choices=[4, 8], value=4, label="Acceleration (R)")
                    mask_radio = gr.Radio(
                        choices=["gaussian", "equispaced"],
                        value="gaussian",
                        label="Mask Type",
                    )
                acs_radio = gr.Radio(choices=[16, 24, 32], value=24, label="ESPIRiT ACS Size")
                cg_steps_slider = gr.Slider(
                    minimum=0, maximum=12, value=6, step=2,
                    label="Medix / MoDL CG Steps (K)",
                )
                tv_iters_slider = gr.Slider(
                    minimum=10, maximum=60, value=40, step=5,
                    label="TV-CS FISTA Iterations",
                )
                err_gain_slider = gr.Slider(
                    minimum=1, maximum=20, value=10, step=1,
                    label="Error Map Amplification (x)",
                )
                supp_mask_cb = gr.Checkbox(
                    value=False,
                    label="Apply Coil-Support Mask to Medix SSDU (suppresses null-space background drift)",
                )

            run_btn = gr.Button("Run Reconstruction & Compare", variant="primary", size="lg")

        with gr.Column(scale=2, min_width=560):
            status_out = gr.HTML(
                '<div class="status-banner"><b>Status:</b> Ready. Upload a scan, enter a Disk/Drive path, or load the synthetic phantom on the left.</div>'
            )
            kpi_out = gr.HTML(build_idle_kpi_html())

            with gr.Tabs():
                with gr.Tab("1. Quantitative Metrics & Export"):
                    metrics_table = gr.Dataframe(
                        label="Benchmark Leaderboard",
                        interactive=False,
                    )
                    with gr.Row():
                        csv_download = gr.File(label="Download Metrics Table (.csv)")
                        zip_download = gr.File(label="Download Full Results Bundle (.zip: PNGs, NPYs, Plots & JSON)")
                    metrics_plot = gr.Plot(label="Quality (PSNR / HFEN) vs. Speed")

                with gr.Tab("2. Interactive A vs. B Slider"):
                    with gr.Row():
                        left_dd = gr.Dropdown(label="Left Image (A)", choices=[])
                        right_dd = gr.Dropdown(label="Right Image (B)", choices=[])
                    compare_slider = gr.ImageSlider(label="Drag Handle to Compare (A vs. B)")
                    ab_diff_plot = gr.Plot(label="Absolute Difference Map |A - B|")

                with gr.Tab("3. Unified Error Maps"):
                    error_grid_plot = gr.Plot(label="Reconstructions & Unified Absolute Error Maps")

                with gr.Tab("4. Synchronized ROI Zoom"):
                    with gr.Row():
                        roi_x_sl = gr.Slider(0, 100, value=35, step=1, label="ROI X (%)")
                        roi_y_sl = gr.Slider(0, 100, value=35, step=1, label="ROI Y (%)")
                        roi_sz_sl = gr.Slider(32, 128, value=64, step=8, label="ROI Box Size (px)")
                    roi_plot = gr.Plot(label="Synchronized Zoomed Anatomical ROI")

                with gr.Tab("5. Physics Diagnostics"):
                    phys_plot = gr.Plot(label="Undersampled k-Space, 1D Sampling Mask & ESPIRiT Maps")

                with gr.Tab("6. Linear Algebra & CG"):
                    with gr.Row():
                        la_model_dd = gr.Dropdown(
                            choices=LA_MODELS,
                            value="Medix (SSDU - T=8)",
                            label="Unrolled Model to Inspect",
                        )
                        la_run_btn = gr.Button("Run Adjoint, Spectral & Krylov Check", variant="secondary")
                    la_summary = gr.HTML(
                        '<div class="status-banner">Verify operator adjointness, spectral norm ||A||_2^2 <= 1, learned mu_t, HPD condition bounds kappa_t, and inner CG residual decay.</div>'
                    )
                    la_plot = gr.Plot(label="Learned mu_t, HPD Bounds & CG Krylov Convergence")
                    la_table = gr.Dataframe(label="Stage-by-Stage Linear-Algebra Table", interactive=False)
                    la_csv_download = gr.File(label="Download Linear-Algebra Table (.csv)")

                with gr.Tab("7. Paper Benchmark & Attribution"):
                    gr.Markdown(
                        """
                        ### Held-Out NYU fastMRI Brain Cohort Highlights (n = 10 Patients, 138 Slices, R = 4x Gaussian)
                        | Method | Supervision | Params (M) | 3D Vol. PSNR (dB) ↑ | Multi-Planar SSIM ↑ | HFEN ↓ | Train VRAM (MB) ↓ |
                        | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
                        | Zero-Filled (ZF) | Non-Learned | 0.000 | 30.52 ± 1.75 | 0.8488 ± 0.0301 | 0.6152 | — |
                        | CG-SENSE (ESPIRiT) | Non-Learned | 0.000 | 31.14 ± 2.66 | 0.7160 ± 0.0672 | 0.4438 | — |
                        | Volumetric TV-CS | Non-Learned | 0.000 | 32.11 ± 3.19 | 0.7453 ± 0.0756 | 0.3543 | — |
                        | fastMRI U-Net | Supervised | 7.756 | 34.68 ± 2.19 | 0.9309 ± 0.0310 | 0.4184 | 934.3 |
                        | fastMRI VarNet | Supervised | 7.030 | 36.10 ± 2.54 | 0.9434 ± 0.0317 | 0.3474 | 3260.7 |
                        | Medix K=0 (No CG) | Supervised | 1.927 | 27.76 ± 1.86 | 0.8197 ± 0.0647 | 0.8154 | 760.5 |
                        | Unrolled-CG / MoDL (UB) | Supervised | 1.927 | 32.95 ± 3.49 | 0.8897 ± 0.0606 | 0.4253 | 746.6 |
                        | Medix (SSDU) | Self-Supervised | 1.927 | 22.45 ± 1.18 | 0.7249 ± 0.0340 | 0.3938 | 767.1 |
                        **Key Findings:**
- Weight-shared **Medix** uses **1.927M parameters** (72.6% fewer than fastMRI VarNet) and **767.1 MB peak training VRAM** (76.5% lower than fastMRI VarNet).
- Explicit differentiable **CG-DC** (`K = 6` vs `K = 0`) improves 3D Volumetric PSNR by **+5.19 dB** and reduces Vol. NMSE by **57.5%**.
- **Medix (SSDU)** achieves the lowest edge error (**HFEN = 0.5183**) across all 8 methods under zero-shot `R = 8x` acceleration; enabling the **Coil-Support Mask** toggle in Section 3 demonstrates suppressing the null-space air-border drift.
                        ---
                        **Dataset Attribution & Non-Commercial Research Notice:**
                        Data used in the preparation of this work were obtained from the **NYU fastMRI Initiative** database (`https://fastmri.med.nyu.edu`). NYU fastMRI investigators provided data but did not participate in the analysis or writing of this work. No patient scans from NYU fastMRI are redistributed in this repository; pretrained weights are shared strictly for non-commercial research and educational use.
                        """
                    )

    btn_phantom.click(fn=on_load_phantom, outputs=[file_in], api_name=False)
    btn_core.click(fn=lambda: CORE_BENCHMARK, outputs=[methods_cb], api_name=False)
    btn_classical.click(fn=lambda: CLASSICAL_METHODS, outputs=[methods_cb], api_name=False)
    btn_all.click(fn=lambda: ALL_METHODS, outputs=[methods_cb], api_name=False)
    btn_clear.click(fn=lambda: [], outputs=[methods_cb], api_name=False)

    file_in.change(
        fn=on_file_change,
        inputs=[file_in, active_file_state],
        outputs=[active_file_state, slice_slider, file_info],
        api_name=False,
    )
    btn_load_local.click(
        fn=on_load_local_path,
        inputs=[local_path_in],
        outputs=[active_file_state, slice_slider, file_info],
        api_name=False,
    )

    run_btn.click(
        fn=run_benchmark,
        api_name="reconstruct",
        inputs=[
            active_file_state,
            file_in,
            local_path_in,
            slice_slider,
            methods_cb,
            accel_radio,
            mask_radio,
            acs_radio,
            cg_steps_slider,
            tv_iters_slider,
            supp_mask_cb,
            err_gain_slider,
            roi_x_sl,
            roi_y_sl,
            roi_sz_sl,
        ],
        outputs=[
            results_state,
            status_out,
            kpi_out,
            metrics_table,
            csv_download,
            zip_download,
            metrics_plot,
            left_dd,
            right_dd,
            compare_slider,
            ab_diff_plot,
            error_grid_plot,
            roi_plot,
            phys_plot,
        ],
    )

    la_run_btn.click(
        fn=run_la_tab,
        api_name="la_diagnostics",
        inputs=[
            active_file_state,
            file_in,
            local_path_in,
            slice_slider,
            accel_radio,
            mask_radio,
            acs_radio,
            cg_steps_slider,
            la_model_dd,
        ],
        outputs=[la_summary, la_plot, la_table, la_csv_download],
    )

    left_dd.change(
        fn=on_ab_change, inputs=[results_state, left_dd, right_dd], outputs=[compare_slider, ab_diff_plot], api_name=False
    )
    right_dd.change(
        fn=on_ab_change, inputs=[results_state, left_dd, right_dd], outputs=[compare_slider, ab_diff_plot], api_name=False
    )

    for sl in (roi_x_sl, roi_y_sl, roi_sz_sl):
        sl.release(
            fn=on_roi_change, inputs=[results_state, roi_x_sl, roi_y_sl, roi_sz_sl], outputs=[roi_plot], api_name=False
        )

_queue_params = inspect.signature(demo.queue).parameters
_queue_kwargs = {}
if "max_size" in _queue_params:
    _queue_kwargs["max_size"] = MAX_QUEUE_SIZE
if "default_concurrency_limit" in _queue_params:
    _queue_kwargs["default_concurrency_limit"] = MAX_CONCURRENT_JOBS

if __name__ == "__main__":
    demo.queue(**_queue_kwargs).launch(**_launch_kwargs)
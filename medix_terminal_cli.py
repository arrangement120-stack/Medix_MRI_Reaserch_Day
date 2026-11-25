import argparse
import json
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from medix_engine import (
    ALL_METHODS,
    CLASSICAL_METHODS,
    MRIPhysics,
    create_synthetic_phantom_h5,
    eval_slice_metrics,
    get_device,
    inspect_uploaded_file,
    load_slice_bundle,
    plot_metrics_chart,
    plot_physics_diagnostics,
    plot_unified_error_grid,
    reconstruction_export_files,
    run_single_method,
)

CORE_BENCHMARK = [
    "Zero-Filled (ZF)",
    "CG-SENSE (ESPIRiT)",
    "Medix (SSDU - T=8)",
    "Unrolled-CG / MoDL (Sup UB)",
]


def resolve_methods(method_args):
    if not method_args:
        return CORE_BENCHMARK
    if len(method_args) == 1:
        key = method_args[0].strip().lower()
        if key == "core":
            return CORE_BENCHMARK
        if key == "classical":
            return CLASSICAL_METHODS
        if key == "all":
            return ALL_METHODS
    invalid = [m for m in method_args if m not in ALL_METHODS]
    if invalid:
        raise ValueError(
            f"Unknown method(s): {invalid}\n"
            f"Valid options: {ALL_METHODS}\n"
            f"Or use shortcuts: core | classical | all"
        )
    return method_args


def main():
    parser = argparse.ArgumentParser(
        description="Medix MRI Reconstruction Studio — Direct Terminal CLI"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw NYU fastMRI (.h5) or preprocessed (.npz) file. Omit to run on the built-in synthetic 12-coil phantom.",
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        help="Slice index for multi-slice .h5 files (default: middle slice).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["core"],
        help="Method names or shortcut: core | classical | all (default: core).",
    )
    parser.add_argument("--accel", type=int, choices=[4, 8], default=4, help="Acceleration factor R (default: 4).")
    parser.add_argument(
        "--mask",
        type=str,
        choices=["gaussian", "equispaced"],
        default="gaussian",
        help="1D Cartesian undersampling mask type (default: gaussian).",
    )
    parser.add_argument("--acs", type=int, choices=[16, 24, 32], default=24, help="ESPIRiT ACS size (default: 24).")
    parser.add_argument("--cg-steps", type=int, default=6, help="Inner CG steps K for Medix/MoDL (default: 6).")
    parser.add_argument("--tv-iters", type=int, default=40, help="FISTA iterations for TV-CS (default: 40).")
    parser.add_argument(
        "--support-mask",
        action="store_true",
        help="Apply coil-support mask to Medix SSDU to suppress null-space background drift.",
    )
    parser.add_argument("--err-gain", type=float, default=10.0, help="Error map amplification factor (default: 10).")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Also save diagnostic figures (metrics chart, unified error grid, physics diagnostics) as PNGs.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./medix_cli_out",
        help="Output directory for metrics.csv, run_info.json, and reconstructions (default: ./medix_cli_out).",
    )
    args = parser.parse_args()

    selected_methods = resolve_methods(args.methods)
    fpath = args.input or create_synthetic_phantom_h5()
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Input file not found: {fpath}")

    n_slices, desc = inspect_uploaded_file(fpath)
    slice_idx = (n_slices // 2) if args.slice is None else max(0, min(int(args.slice), n_slices - 1))

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    physics = MRIPhysics()

    print(f"[Medix CLI] Device : {device.type.upper()}")
    print(f"[Medix CLI] Input  : {fpath} ({desc})")
    print(f"[Medix CLI] Slice  : {slice_idx}/{max(n_slices - 1, 0)} | R={args.accel}x ({args.mask}) | ACS={args.acs}")
    print(f"[Medix CLI] Methods: {', '.join(selected_methods)}")

    with torch.inference_mode():
        bundle = load_slice_bundle(
            fpath=fpath,
            slice_idx=slice_idx,
            accel=args.accel,
            mask_type=args.mask,
            acs_size=args.acs,
            device=device,
        )
        gt = bundle["gt"]
        recons = {}
        if gt is not None:
            recons["Ground Truth (RSS)"] = np.asarray(gt, dtype=np.float32)

        rows = []
        for idx, m in enumerate(selected_methods, start=1):
            print(f"  [{idx}/{len(selected_methods)}] Running {m}...")
            rec, dt_ms = run_single_method(
                method_name=m,
                bundle=bundle,
                physics=physics,
                device=device,
                cg_steps_medix=args.cg_steps,
                tv_iters=args.tv_iters,
                apply_support_mask=args.support_mask,
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

    df = pd.DataFrame(rows)
    if df["PSNR (dB) ↑"].notna().any():
        df = df.sort_values("PSNR (dB) ↑", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(args.out, "metrics.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    ref = gt if gt is not None else recons[selected_methods[0]]
    vmax_img = float(np.percentile(ref, 99.7)) + 1e-12
    for rel_path, raw_bytes in reconstruction_export_files(recons, vmax_img).items():
        full_path = os.path.join(args.out, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(raw_bytes)

    if args.save_plots:
        fig_dir = os.path.join(args.out, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        figs = {
            "metrics_chart": plot_metrics_chart(df),
            "unified_error_grid": plot_unified_error_grid(recons, gt, vmax_img, args.err_gain),
            "physics_diagnostics": plot_physics_diagnostics(bundle),
        }
        for name, fig in figs.items():
            fig.savefig(os.path.join(fig_dir, f"{name}.png"), format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device.type.upper(),
        "input_file": os.path.basename(fpath),
        "slice_index": slice_idx,
        "methods": selected_methods,
        "accel": args.accel,
        "mask_type": args.mask,
        "acs_size": args.acs,
        "cg_steps": args.cg_steps,
        "tv_iterations": args.tv_iters,
        "coil_support_mask": bool(args.support_mask),
    }
    with open(os.path.join(args.out, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + df.to_string(index=False))
    print(f"\n[Medix CLI] Saved outputs to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
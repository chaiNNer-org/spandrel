"""Standalone spandrel arch benchmark (no project dependencies).

Loads a .pth via spandrel.ModelLoader and reports, per resolution/dtype:
  - throughput (fps) of the raw forward (steady-state, channels_last)
  - peak VRAM
  - PSNR / SSIM of the fp16 output vs the fp32 output (the "fp16 noise floor")
  - VMAF of the fp16 output vs the fp32 output over a short clip (via ffmpeg
    libvmaf), as a perceptual cross-check of the fp16 noise

Used to validate the PLKSR / NAFNet / SPAN fp16 normalization speedups.

Example:
  python bench_spandrel.py \\
    --model plksr=weights/real-plksr/1xDeJPG_realplksr_otf.pth \\
    --model span=weights/span/2x_ModernSpanimationV2.pth \\
    --frame input/720.mp4 --res 360p 720p 1080p --dtype fp16 fp32 \\
    --vmaf --vmaf-res 720p --json after.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

# Import spandrel before anything that might add a stray "spandrel" dir to path.
from spandrel import ModelLoader  # noqa: E402

try:
    from spandrel_extra_arches import install as _install_extra

    _install_extra(ignore_duplicates=True)
except Exception:
    pass

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

RES = {"360p": (640, 360), "720p": (1280, 720), "1080p": (1920, 1080)}
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def frames_from(path, width, height, n, in_nc=3):
    """Yield up to n NCHW [0,1] cuda tensors decoded+resized from a video."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    got = 0
    while got < n:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.cvtColor(cv2.resize(img, (width, height)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).cuda().permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if in_nc > 3:
            t = torch.cat([t, t.new_full((1, in_nc - 3, height, width), 15 / 255)], 1)
        yield t
        got += 1
    cap.release()


def load(path):
    desc = ModelLoader().load_from_file(path)
    in_nc = 3
    for m in desc.model.modules():
        if isinstance(m, torch.nn.Conv2d):
            in_nc = m.in_channels
            break
    arch = getattr(getattr(desc, "architecture", None), "name", "?")
    return desc.model.eval().cuda(), getattr(desc, "scale", 1), arch, in_nc


@torch.inference_mode()
def perf(model, inp, frames, warmup=10):
    torch.cuda.synchronize()
    for _ in range(warmup):
        model(inp)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(frames):
        model(inp)
    e.record()
    torch.cuda.synchronize()
    fps = frames * 1000.0 / s.elapsed_time(e)
    return fps, torch.cuda.max_memory_reserved() / 1024**2


@torch.inference_mode()
def quality(model, frame32):
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio as psnr,
        structural_similarity_index_measure as ssim,
    )

    ref = model.float()(frame32.to(memory_format=torch.channels_last)).float().clamp(0, 1)
    out = (
        model.half()(frame32.half().to(memory_format=torch.channels_last))
        .float()
        .clamp(0, 1)
    )
    return psnr(out, ref, data_range=1.0).item(), ssim(out, ref, data_range=1.0).item()


def _write_pngs(model, dtype, frames, out_dir, tag):
    paths = []
    with torch.inference_mode():
        m = model.to(dtype)
        for i, f in enumerate(frames):
            o = m(f.to(dtype).to(memory_format=torch.channels_last)).float().clamp(0, 1)
            img = (o[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            p = os.path.join(out_dir, f"{tag}_{i:04d}.png")
            cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            paths.append(p)
    return paths


def vmaf(model, src, width, height, n, in_nc):
    """VMAF of fp16 output vs fp32 output over n frames."""
    with tempfile.TemporaryDirectory() as td:
        f32 = list(frames_from(src, width, height, n, in_nc))
        _write_pngs(model, torch.float32, f32, td, "ref")
        _write_pngs(model, torch.float16, f32, td, "dis")
        cmd = [
            "ffmpeg", "-hide_banner", "-y",
            "-framerate", "25", "-i", os.path.join(td, "dis_%04d.png"),
            "-framerate", "25", "-i", os.path.join(td, "ref_%04d.png"),
            "-lavfi", "libvmaf", "-f", "null", "-",
        ]
        out = subprocess.run(cmd, capture_output=True, text=True).stderr
        for line in out.splitlines():
            if "VMAF score" in line:
                return float(line.split("VMAF score:")[1].strip())
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=path")
    ap.add_argument("--res", nargs="+", default=["360p", "720p", "1080p"])
    ap.add_argument("--dtype", nargs="+", default=["fp16", "fp32"])
    ap.add_argument("--frame", default="input/720.mp4")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--vmaf", action="store_true")
    ap.add_argument("--vmaf-res", default="720p")
    ap.add_argument("--vmaf-frames", type=int, default=48)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("medium")
    rows = []
    for spec in args.model:
        name, path = spec.split("=", 1)
        model, scale, arch, in_nc = load(path)
        print(f"\n=== {name} (arch={arch}, scale={scale}, in_nc={in_nc}) ===")
        for res in args.res:
            w, h = RES[res]
            f32 = next(frames_from(args.frame, w, h, 1, in_nc))
            try:
                p, s = quality(model, f32)
            except Exception as e:
                p, s = float("nan"), float("nan")
                print(f"  [{res}] quality fail: {e!r}")
            for dt in args.dtype:
                if dt == "fp32" and res == "1080p":
                    continue  # fp32 only at 360/720 per protocol
                model.to(DTYPES[dt]).to(memory_format=torch.channels_last)
                inp = (
                    next(frames_from(args.frame, w, h, 1, in_nc))
                    .to(DTYPES[dt])
                    .contiguous()
                    .to(memory_format=torch.channels_last)
                )
                with torch.inference_mode():
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(5):
                        model(inp)
                    torch.cuda.synchronize()
                per = (time.perf_counter() - t0) / 5
                nf = max(20, min(args.cap, int(args.seconds / max(per, 1e-6))))
                fps, vram = perf(model, inp, nf)
                v = float("nan")
                if args.vmaf and dt == "fp16" and res == args.vmaf_res:
                    v = vmaf(model, args.frame, w, h, args.vmaf_frames, in_nc)
                rows.append({
                    "model": name, "arch": arch, "res": res, "dtype": dt,
                    "fps": round(fps, 2), "vram_mb": round(vram, 0),
                    "psnr": round(p, 2), "ssim": round(s, 5),
                    "vmaf": round(v, 3) if v == v else None,
                })
                print(
                    f"  [{res}/{dt}] fps={fps:8.2f} vram={vram:7.0f}MB "
                    f"PSNR={p:6.2f} SSIM={s:.5f}"
                    + (f" VMAF={v:.3f}" if v == v else "")
                )
        del model
        torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

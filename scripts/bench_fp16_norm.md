# fp16 normalization speedups — benchmarks

Benchmarks for the PLKSR / NAFNet / SPAN fp16 inference fixes. Measured with
`scripts/bench_fp16_norm.py` on an **NVIDIA RTX 3090**, PyTorch 2.12 + CUDA,
`channels_last`, steady-state forward throughput. Source frames decoded from a
1080p anime clip and resized per resolution.

Quality columns compare the **fp16 output against the fp32 output** of the same
model (the "fp16 noise floor"): PSNR / SSIM on a frame, plus VMAF over a short
clip via ffmpeg `libvmaf`.

## PLKSR (`real-plksr`, 1x)

| res | dtype | fps before | fps after | Δ | PSNR | SSIM | VMAF |
|-----|-------|-----------:|----------:|----:|-----:|------|-----:|
| 360p  | fp16 | 5.99 | **7.83** | +31% | 79.6 | 0.99999 | — |
| 360p  | fp32 | 3.63 | **4.31** | +19% | 79.6 | 0.99999 | — |
| 720p  | fp16 | 1.47 | **1.92** | +31% | 79.3 | 0.99999 | 98.1 |
| 720p  | fp32 | 0.89 | **1.06** | +19% | 79.3 | 0.99999 | — |
| 1080p | fp16 | 0.63 | **0.84** | +33% | 79.4 | 0.99999 | — |

PSNR/SSIM/VMAF unchanged — the fix moves the GroupNorm statistics into fp32, so
fp16 output is numerically equal to (or slightly better than) before. Reserved
VRAM unchanged.

## NAFNet (`NAFNet-GoPro-width64`, 1x)

fp16 was **broken** on the original arch (LayerNorm overflow → garbage output,
PSNR ~4 dB, VMAF 0), so the only usable mode was fp32. The fix makes fp16
correct; comparison is **prev-usable fp32 → now-correct fp16**:

| res | mode | fps | PSNR | SSIM | VMAF |
|-----|------|----:|-----:|------|-----:|
| 360p  | fp32 (before, usable) | 15.4 | — | — | — |
| 360p  | fp16 (after, correct) | **22.6** (+47%) | 75.9 | 0.99999 | — |
| 720p  | fp32 (before, usable) | 4.08 | — | — | — |
| 720p  | fp16 (after, correct) | **5.97** (+46%) | 76.6 | 0.99999 | 98.2 |
| 1080p | fp16 (after, correct) | **2.65** | 80.6 | 0.99999 | — |

Original fp16 for reference: PSNR 4.2 dB, SSIM 0.007, VMAF 0.0 (unusable).

## SPAN (`2x_ModernSpanimationV2`, 2x)

| res | dtype | fps before | fps after | Δ | PSNR | SSIM | VMAF |
|-----|-------|-----------:|----------:|----:|-----:|------|-----:|
| 360p  | fp16 | 82.6 | **104.8** | +27% | 62.9 | 0.9996 | — |
| 360p  | fp32 | 39.2 | **44.7**  | +14% | 62.9 | 0.9996 | — |
| 720p  | fp16 | 24.9 | **26.8**  | +7%  | 61.7 | 0.9995 | 98.2 |
| 720p  | fp32 | 11.1 | **11.6**  | +4%  | 61.7 | 0.9995 | — |
| 1080p | fp16 | 11.7 | **12.1**  | +4%  | 61.5 | 0.9994 | — |

SPAN output is bit-identical before/after (the change only removes a per-forward
recomputation); PSNR/SSIM/VMAF reflect SPAN's inherent fp16 sensitivity, equal
in both. Gains concentrate at low resolution where the per-frame reparam
overhead dominated.

## Reproduce

```
python scripts/bench_fp16_norm.py \
  --model real-plksr=/path/1xDeJPG_realplksr_otf.pth \
  --model nafnet=/path/NAFNet-GoPro-width64.pth \
  --model span=/path/2x_ModernSpanimationV2.pth \
  --frame /path/clip.mp4 --res 360p 720p 1080p --dtype fp16 fp32 \
  --vmaf --vmaf-res 720p
```

VMAF requires an `ffmpeg` build with `libvmaf`. fps is steady-state rate, so it
is comparable across runs regardless of the sampled frame count.

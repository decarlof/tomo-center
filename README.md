# tomo-center-ai

Standalone AI rotation-axis center picker for pre-reconstructed TIFF slices.

The model and inference code are **vendored from
[tomocupy/develop](https://github.com/stang292/tomocupy/tree/develop/src/tomocupy/ai)**
(BSD-3, UChicago Argonne LLC). This repo packages just the bits needed to run
the classifier outside the full `tomocupy` reconstruction stack — no CUDA build,
no SWIG, no HDF5 reader.

## What it does

Given a folder of TIFF slices — each one already reconstructed at a different
candidate center of rotation — the DINOv2-based classifier scores every slice
and writes the candidate(s) with the highest "correct center" probability.

Input shape per slice: 2D, any dtype (cast to float32). All slices must share
the same `(H, W)`.

## Install

Use a dedicated conda env — `torch` and a pinned `numpy<2` would conflict with
most existing envs.

```bash
conda create -n tomo-center-ai python=3.10 pip -y
conda activate tomo-center-ai
pip install -e /path/to/tomo-center-ai
```

The editable install (`pip install -e .`) pulls every runtime dep declared in
`pyproject.toml`. The full list:

| Package    | Why                                                          |
| ---------- | ------------------------------------------------------------ |
| `numpy<2`  | array ops; pinned because `torch 2.2.x` wheels were built against NumPy 1.x |
| `pillow`   | image resize (PIL `Image.fromarray` / `BILINEAR`) inside the inference pipeline |
| `tifffile` | reading TIFF slices from the input folder                    |
| `torch`    | DINOv2 backbone + classifier head                            |
| `einops`   | tensor rearrange used inside `model_archs.py`                |

GPU inference is automatic when CUDA is available — install the matching
`torch` CUDA wheel for your system (see https://pytorch.org/get-started). CPU
works but is slow.

## Get the model checkpoint

```
https://anl.box.com/s/4o8qcig6pl9k8p7x4z3qqbrpgnjipolq
```

## Run

```bash
tomo-center-ai /path/to/recons \
    --model-path /path/to/model.pt \
    --out-dir   /path/to/out
```

### Example

200 slices reconstructed at centers 974.5–1074.0, run on tomo4 (GPU):

```console
$ tomo-center-ai \
    /data2/2BM/2023-04/Strumendo-2023-04_rec/try_center/CaCO3room_001/ \
    --model-path /home/beams/2BMB/models/datav2_518_full_finetune.pt \
    --out-dir    ~/tomo-center-ai-out/CaCO3room_001 \
    --plot
2026-06-05 20:03:03,134 - Loading TIFFs from /data2/2BM/2023-04/Strumendo-2023-04_rec/try_center/CaCO3room_001 ...
2026-06-05 20:03:05,058 -   200 slices, shape (2048, 2048), dtype float32
2026-06-05 20:03:05,059 -   center range: 974.5 .. 1074.0
2026-06-05 20:03:06,894 - starting model inference...
2026-06-05 20:03:06,894 - Downsample factor is 1. No resizing applied.
2026-06-05 20:03:11,113 - done. Elapsed time is 4.22 s.
2026-06-05 20:03:11,121 - Best center(s):
2026-06-05 20:03:11,121 -   1023.0
```

Score curve written to `<out-dir>/scores.png`:

![Score curve example](docs/source/img/scores.png)

### How centers are paired with TIFFs

By default the **last numeric token in each filename stem** is used as that
slice's candidate center, e.g. `recon_0001_1234.50.tif → 1234.5`.

If your filenames don't carry the center, supply a sidecar file:

```bash
tomo-center-ai /path/to/recons \
    --model-path /path/to/model.pt \
    --centers-file centers.txt
```

`centers.txt` is one float per line, in the same sorted order as the TIFFs.
Lines starting with `#` are ignored.

### Multi-scale inference

The upstream pipeline supports running multiple `(downsample, num_windows, window_size)`
scales and combining their features. Pass matching-length lists:

```bash
tomo-center-ai /path/to/recons --model-path model.pt \
    --downsample-factor 1 2 \
    --num-windows       4 4 \
    --window-size       224 224
```

## Output

- `center_of_rotation.txt` — one center per line (appended, not overwritten).
- `predicts_all.npz` — raw model logits + the center list, if
  `--save-intermediate` is passed.
- `scores.png` (or a custom path) — per-slice probability vs. candidate
  center, if `--plot` is passed.

## Diagnostic plot

```bash
pip install -e '.[plot]'                # adds matplotlib

tomo-center-ai /path/to/recons \
    --model-path /path/to/model.pt \
    --plot                              # → <out-dir>/scores.png
# or:
tomo-center-ai /path/to/recons --model-path model.pt --plot my-scan.png
```

A sharp peak with neighbors tapering off → confident pick (see the
[example](#example) above). A flat curve or several near-ties at the top → the
sweep was too coarse, too narrow, or the slices don't carry enough signal for
the classifier to discriminate; re-sweep finer around the picked value and run
again.

## Attribution

- `src/tomo_center_ai/ai/inference.py` — vendored from
  `tomocupy/src/tomocupy/ai/inference.py`. Changes vs. upstream: internal
  import path; switched `print()` to the package logger; fixed an
  `UnboundLocalError` in the single-instance branch (`patch_corner` →
  `patch_corners`).
- `src/tomo_center_ai/ai/model_archs.py` — vendored verbatim from
  `tomocupy/src/tomocupy/ai/model_archs.py`. It in turn includes a DINOv2 ViT
  (Apache-2.0, Meta) and attention pooling (MIT, Ilse & Tomczak).
- `src/tomo_center_ai/logging.py` — adapted from
  `tomocupy/src/tomocupy/logging.py` (same colored-console formatter, scoped
  to the `tomo_center_ai.*` logger tree).
- See `LICENSE` for the upstream BSD-3 terms.

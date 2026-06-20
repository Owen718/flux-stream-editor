# Realtime Editing Fast Module

This folder keeps a reusable, cleaner FLUX.2 realtime editing wrapper without changing `flux2.py`.

## Files

- `editor.py`: `FastFlux2RealtimeEditor` core wrapper.
- `realtime_txt2img_server.py`: FastAPI realtime txt2img demo server.
- `realtime_img2img_server.py`: FastAPI realtime img2img demo server.
- `static/realtime_txt2img/index.html`: 4x4 realtime txt2img frontend.
- `static/realtime_img2img/index.html`: webcam/screen realtime img2img frontend.
- `__init__.py`: package export.

## Config

Default config inside `FastFlux2Config` targets **HD single-shot quality**, not realtime throughput:

- `1280x720` (HD, configurable via `--width` / `--height`)
- `28-step`
- `DBCache + TaylorSeer` are **disabled** by default (max fidelity)
- `torch.compile` for transformer/VAE is **disabled** by default
- `TAEF2` tiny VAE is **disabled** by default (full VAE for HD)

If you want to trade quality for throughput, you can flip pieces back on via env vars
(`FLUX_ENABLE_CACHE_DIT=1`, `FLUX_ENABLE_TAYLORSEER=1`, `FLUX_COMPILE_TRANSFORMER=1`,
`FLUX_VAE_ENCODE_COMPILE=1`, `FLUX_VAE_DECODE_COMPILE=1`, `FLUX_USE_TAEF2=1`).
Note: when `FLUX_ENABLE_CACHE_DIT=1`, make sure `steps_mask` length matches
`num_inference_steps`.

## Gradio Entry

Use root script:

```bash
python gradio_realtime_editing.py --host 0.0.0.0 --port 7860
```

## FastAPI Realtime Txt2Img Entry

Run a StreamDiffusion-style realtime txt2img demo (without reusing StreamDiffusion code):

```bash
uv pip install fastapi uvicorn
python -m realtime_editing_fast.realtime_txt2img_server --host 0.0.0.0 --port 7860
```

Then open `http://localhost:7860`.

## FastAPI Realtime Img2Img Entry

Run a StreamDiffusion-style realtime img2img demo:

```bash
uv pip install fastapi uvicorn
python -m realtime_editing_fast.realtime_img2img_server --host 0.0.0.0 --port 7870 --num-inference-steps 2
```

Then open `http://localhost:7870`.

## Video Upload Endpoint (img2img)

The img2img server also exposes a synchronous video editing endpoint that edits
every frame with a frozen prompt/seed for style consistency, then muxes the
result back to MP4 at the source FPS.

Install deps:

```bash
uv pip install imageio imageio-ffmpeg
```

Request:

```bash
curl -X POST "http://localhost:7870/api/predict_video" \
  -F "video=@input.mp4" \
  -F "prompt=Cinematic anime illustration, clean lines, rich color" \
  -F "seed=42" \
  -o edited.mp4
```

Notes:

- `seed=-1` randomizes once and reuses the same seed for the whole video (still consistent across frames).
- Supported extensions: `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, `.m4v`.
- Max upload size defaults to 512 MiB; override via `FLUX_VIDEO_MAX_BYTES`.
- Response headers include `x-flux-frame-count`, `x-flux-src-fps`, `x-flux-wall-seconds`, `x-flux-avg-frame-total-ms`.
- Endpoint is synchronous and serialized behind the same model lock as `/api/predict`. For long videos, expect long HTTP wait times — keep client timeouts generous.

## TAEF2 Switch

Enable TAEF2 VAE for realtime acceleration:

```bash
FLUX_USE_TAEF2=1 python -m realtime_editing_fast.realtime_img2img_server --host 0.0.0.0 --port 7870 --num-inference-steps 2
```

Optional env knobs:

- `FLUX_TAEF2_FORCE_EAGER_VAE` (default `1`): disable VAE encode/decode compile when TAEF2 is enabled.
- `FLUX_TAEF2_CACHE_DIR` (default `.cache/taef2`): cache path for `taesd.py` and `taef2.safetensors`.
- `FLUX_TAEF2_SCRIPT_PATH`: explicit local path for `taesd.py`.
- `FLUX_TAEF2_WEIGHT_PATH`: explicit local path for `taef2.safetensors`.

If `gradio` is missing:

```bash
uv pip install gradio
```

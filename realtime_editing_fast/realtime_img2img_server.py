from __future__ import annotations

import argparse
import asyncio
import base64
import os
import secrets
import shutil
import tempfile
import time
import uuid
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from .editor import FastFlux2Config, FastFlux2RealtimeEditor, normalize_attention_backend_name


DEFAULT_PROMPT = "Convert this live frame into a cinematic anime illustration with clean lines and rich color."
ATTENTION_BACKEND_CHOICES = ["auto", "sage", "native", "none", "sage_hub", "_flash_3", "fa3"]
STATIC_DIR = Path(__file__).resolve().parent / "static" / "realtime_img2img"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class LoadInputModel(BaseModel):
    attention_backend: str | None = Field(default=None, description="Use server default if not provided")


class LoadResponseModel(BaseModel):
    status: str
    attention_backend: str


class PredictInputModel(BaseModel):
    base64_image: str
    prompt: str = Field(default=DEFAULT_PROMPT)
    seed: int = Field(default=0, description="Use -1 for random seed")


class PredictResponseModel(BaseModel):
    base64_image: str
    seed: int
    request_tag: str
    total_ms: float
    refresh_ms: float
    prepare_ms: float
    decode_ms: float
    source_size: tuple[int, int]
    target_size: tuple[int, int]


# Max upload size for video endpoint (in bytes). Override with FLUX_VIDEO_MAX_BYTES.
DEFAULT_VIDEO_MAX_BYTES = 512 * 1024 * 1024  # 512 MiB
VIDEO_MAX_BYTES = int(os.getenv("FLUX_VIDEO_MAX_BYTES", str(DEFAULT_VIDEO_MAX_BYTES)))
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


class HealthResponseModel(BaseModel):
    status: str
    model_loaded: bool


class SettingsResponseModel(BaseModel):
    default_prompt: str
    width: int
    height: int
    num_inference_steps: int


class GPUInfoResponseModel(BaseModel):
    device_name: str
    device_count: int
    cuda_available: bool
    cuda_version: str | None


class RealtimeImg2ImgApi:
    def __init__(self, config: FastFlux2Config) -> None:
        self.editor = FastFlux2RealtimeEditor(config)
        self.app = FastAPI(title="Flux2 Realtime Img2Img")
        self._model_lock = asyncio.Lock()
        self._setup_routes()

    @staticmethod
    def _normalize_attention_backend(attention_backend: str) -> str:
        return normalize_attention_backend_name(attention_backend)

    @staticmethod
    def _resolve_seed(seed: int) -> int:
        if int(seed) >= 0:
            return int(seed)
        return secrets.randbelow(2**31 - 1)

    @staticmethod
    def _pil_to_base64(image: Image.Image, format: str = "JPEG") -> str:
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format=format, quality=92)
        return base64.b64encode(buffered.getvalue()).decode("ascii")

    @staticmethod
    def _base64_to_pil(base64_image: str) -> Image.Image:
        encoded = base64_image
        if "," in encoded and "base64" in encoded[:40].lower():
            encoded = encoded.split(",", 1)[1]
        data = base64.b64decode(encoded)
        return Image.open(BytesIO(data)).convert("RGB")

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass

    @staticmethod
    def _safe_rmtree(path: Path) -> None:
        try:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

    def _process_video_sync(
        self,
        input_video_path: Path,
        output_video_path: Path,
        prompt: str,
        seed: int,
    ) -> dict:
        """Read frames sequentially, edit each one with a frozen seed/prompt for
        style consistency, then mux back to MP4 at the source FPS.

        Returns a metadata dict with timings and frame info.
        """
        try:
            import imageio  # type: ignore
            import imageio_ffmpeg  # noqa: F401  # ensures ffmpeg backend is installed
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "imageio + imageio-ffmpeg are required for video processing. "
                    "Install via `uv pip install imageio imageio-ffmpeg`."
                ),
            ) from exc

        # Probe source FPS via the ffmpeg reader metadata.
        reader = imageio.get_reader(str(input_video_path), "ffmpeg")
        try:
            src_meta = reader.get_meta_data() or {}
            src_fps = float(src_meta.get("fps") or 24.0)
            if src_fps <= 0 or not np.isfinite(src_fps):
                src_fps = 24.0

            t_start = time.perf_counter()
            frame_count = 0
            sum_total_ms = 0.0

            writer = imageio.get_writer(
                str(output_video_path),
                fps=src_fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=1,
                quality=None,
                ffmpeg_log_level="error",
                output_params=["-crf", "18", "-preset", "medium", "-movflags", "+faststart"],
            )
            try:
                for frame in reader:
                    pil_frame = Image.fromarray(np.asarray(frame)).convert("RGB")

                    edited, frame_meta = self.editor.edit_image_with_meta(
                        image=pil_frame,
                        prompt=prompt,
                        seed=seed,
                    )

                    out_arr = np.asarray(edited.convert("RGB"))
                    # H.264 yuv420p requires even width/height; pad by 1px if odd.
                    h, w = out_arr.shape[:2]
                    pad_h = h % 2
                    pad_w = w % 2
                    if pad_h or pad_w:
                        out_arr = np.pad(
                            out_arr,
                            ((0, pad_h), (0, pad_w), (0, 0)),
                            mode="edge",
                        )
                    writer.append_data(out_arr)

                    sum_total_ms += float(frame_meta.get("total_ms", 0.0))
                    frame_count += 1
            finally:
                writer.close()
        finally:
            reader.close()

        elapsed_s = time.perf_counter() - t_start
        return {
            "frame_count": frame_count,
            "src_fps": src_fps,
            "wall_seconds": elapsed_s,
            "avg_frame_total_ms": (sum_total_ms / frame_count) if frame_count else 0.0,
        }

    def _setup_routes(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/api/health", response_model=HealthResponseModel)
        async def health() -> HealthResponseModel:
            return HealthResponseModel(status="ok", model_loaded=self.editor.is_loaded)

        @self.app.get("/api/settings", response_model=SettingsResponseModel)
        async def settings() -> SettingsResponseModel:
            cfg = self.editor.config
            return SettingsResponseModel(
                default_prompt=DEFAULT_PROMPT,
                width=int(cfg.width),
                height=int(cfg.height),
                num_inference_steps=int(cfg.num_inference_steps),
            )

        @self.app.get("/api/gpu_info", response_model=GPUInfoResponseModel)
        async def gpu_info() -> GPUInfoResponseModel:
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            device_name = "N/A"
            cuda_version = None
            
            if cuda_available and device_count > 0:
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
            
            return GPUInfoResponseModel(
                device_name=device_name,
                device_count=device_count,
                cuda_available=cuda_available,
                cuda_version=cuda_version,
            )

        @self.app.post("/api/load", response_model=LoadResponseModel)
        async def load_model(inp: LoadInputModel) -> LoadResponseModel:
            async with self._model_lock:
                backend_from_input = inp.attention_backend or self.editor.config.attention_backend
                selected_backend = self._normalize_attention_backend(backend_from_input)
                if selected_backend not in ATTENTION_BACKEND_CHOICES:
                    selected_backend = self.editor.config.attention_backend

                if selected_backend != self.editor.config.attention_backend:
                    self.editor = FastFlux2RealtimeEditor(
                        replace(
                            self.editor.config,
                            attention_backend=selected_backend,
                        )
                    )

                self.editor.ensure_loaded()
                return LoadResponseModel(status="loaded", attention_backend=self.editor.config.attention_backend)

        @self.app.post("/api/predict", response_model=PredictResponseModel)
        async def predict(inp: PredictInputModel) -> PredictResponseModel:
            prompt = (inp.prompt or "").strip() or DEFAULT_PROMPT
            seed = self._resolve_seed(inp.seed)
            frame = self._base64_to_pil(inp.base64_image)

            async with self._model_lock:
                edited, meta = self.editor.edit_image_with_meta(image=frame, prompt=prompt, seed=seed)

            return PredictResponseModel(
                base64_image=self._pil_to_base64(edited),
                seed=seed,
                request_tag=meta["request_tag"],
                total_ms=float(meta["total_ms"]),
                refresh_ms=float(meta["refresh_ms"]),
                prepare_ms=float(meta["prepare_ms"]),
                decode_ms=float(meta["decode_ms"]),
                source_size=tuple(meta["source_size"]),
                target_size=tuple(meta["target_size"]),
            )

        @self.app.post("/api/predict_video")
        async def predict_video(
            background_tasks: BackgroundTasks,
            video: UploadFile = File(..., description="Source video file (mp4/mov/mkv/webm/avi/m4v)."),
            prompt: str = Form(default=DEFAULT_PROMPT),
            seed: int = Form(default=0, description="Fixed seed across all frames keeps style consistent. Use -1 for random (still fixed for the whole video)."),
        ):
            """Edit every frame of an uploaded video with a frozen prompt/seed
            for style consistency, then mux back to MP4 at the source FPS.

            Returns the resulting MP4 file. Metadata (frame_count, fps, timings)
            is returned via response headers (`x-flux-...`).
            """
            # Validate filename / extension.
            original_name = (video.filename or "input").strip()
            suffix = Path(original_name).suffix.lower()
            if suffix not in SUPPORTED_VIDEO_SUFFIXES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported video extension '{suffix}'. Supported: {sorted(SUPPORTED_VIDEO_SUFFIXES)}",
                )

            prompt = (prompt or "").strip() or DEFAULT_PROMPT
            resolved_seed = self._resolve_seed(seed)

            # Stream upload to disk, enforcing max size.
            workdir = Path(tempfile.mkdtemp(prefix="flux_video_"))
            input_path = workdir / f"input{suffix}"
            output_path = workdir / "edited.mp4"

            total_bytes = 0
            try:
                with input_path.open("wb") as f:
                    while True:
                        chunk = await video.read(1024 * 1024)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        if total_bytes > VIDEO_MAX_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"Video too large (> {VIDEO_MAX_BYTES} bytes).",
                            )
                        f.write(chunk)
            except HTTPException:
                self._safe_rmtree(workdir)
                raise
            except Exception as exc:
                self._safe_rmtree(workdir)
                raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc!r}") from exc
            finally:
                await video.close()

            if total_bytes == 0:
                self._safe_rmtree(workdir)
                raise HTTPException(status_code=400, detail="Empty upload.")

            # The pipeline is not thread-safe with respect to text-encode caching
            # and CUDA stream usage; serialize with the same model lock used by
            # /api/predict. We also offload the blocking, GPU-heavy loop to a
            # worker thread so the event loop stays responsive.
            async with self._model_lock:
                try:
                    self.editor.ensure_loaded()
                    video_meta = await asyncio.to_thread(
                        self._process_video_sync,
                        input_path,
                        output_path,
                        prompt,
                        resolved_seed,
                    )
                except HTTPException:
                    self._safe_rmtree(workdir)
                    raise
                except Exception as exc:
                    self._safe_rmtree(workdir)
                    raise HTTPException(status_code=500, detail=f"Video processing failed: {exc!r}") from exc

            if not output_path.exists() or output_path.stat().st_size == 0:
                self._safe_rmtree(workdir)
                raise HTTPException(status_code=500, detail="No output produced.")

            # Cleanup tempdir after the response is fully sent.
            background_tasks.add_task(self._safe_rmtree, workdir)

            download_name = f"{Path(original_name).stem or 'edited'}_flux.mp4"
            headers = {
                "x-flux-seed": str(resolved_seed),
                "x-flux-frame-count": str(int(video_meta["frame_count"])),
                "x-flux-src-fps": f"{video_meta['src_fps']:.3f}",
                "x-flux-wall-seconds": f"{video_meta['wall_seconds']:.3f}",
                "x-flux-avg-frame-total-ms": f"{video_meta['avg_frame_total_ms']:.2f}",
            }
            return FileResponse(
                path=str(output_path),
                media_type="video/mp4",
                filename=download_name,
                headers=headers,
            )

        if not STATIC_DIR.exists():
            raise RuntimeError(f"Static frontend directory not found: {STATIC_DIR}")

        self.app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")


def build_default_config(
    attention_backend: str = "auto",
    num_inference_steps: int = 28,
    width: int = 1280,
    height: int = 720,
) -> FastFlux2Config:
    """Build a non-realtime HD config.

    Cache/compile/TaylorSeer/TAEF2 are OFF by default for max fidelity. You can
    flip them back on via env vars if you want to trade quality for throughput.
    """
    attention_backend = normalize_attention_backend_name(attention_backend)
    profile_stage_timing = os.getenv("FLUX_PROFILE_STAGE", "0") == "1"

    enable_cache_dit = _env_bool("FLUX_ENABLE_CACHE_DIT", False)
    enable_taylorseer = _env_bool("FLUX_ENABLE_TAYLORSEER", False)
    compile_transformer = _env_bool("FLUX_COMPILE_TRANSFORMER", False)
    enable_vae_decoder_compile = _env_bool("FLUX_VAE_DECODE_COMPILE", False)
    vae_decoder_compile_disable_cudagraphs = _env_bool("FLUX_VAE_DECODE_DISABLE_CUDAGRAPHS", True)
    vae_decoder_channels_last = _env_bool("FLUX_VAE_DECODE_CHANNELS_LAST", False)
    vae_decoder_input_channels_last = _env_bool("FLUX_VAE_DECODE_INPUT_CHANNELS_LAST", False)
    vae_decoder_compile_mode = os.getenv("FLUX_VAE_DECODE_COMPILE_MODE", "reduce-overhead").strip() or "reduce-overhead"
    enable_vae_encoder_compile = _env_bool("FLUX_VAE_ENCODE_COMPILE", False)
    vae_encoder_compile_disable_cudagraphs = _env_bool("FLUX_VAE_ENCODE_DISABLE_CUDAGRAPHS", True)
    vae_encoder_compile_mode = os.getenv("FLUX_VAE_ENCODE_COMPILE_MODE", "reduce-overhead").strip() or "reduce-overhead"
    enable_taef2 = _env_bool("FLUX_USE_TAEF2", False)
    taef2_force_eager_vae = _env_bool("FLUX_TAEF2_FORCE_EAGER_VAE", True)
    taef2_cache_dir = os.getenv("FLUX_TAEF2_CACHE_DIR", ".cache/taef2").strip() or ".cache/taef2"
    taef2_taesd_py_path = os.getenv("FLUX_TAEF2_SCRIPT_PATH", "").strip()
    taef2_weight_path = os.getenv("FLUX_TAEF2_WEIGHT_PATH", "").strip()
    cache_timesteps = _env_bool("FLUX_CACHE_TIMESTEPS", False)
    cache_image_latent_ids = _env_bool("FLUX_CACHE_IMAGE_LATENT_IDS", False)

    # steps_mask is only consumed when cache-dit is enabled; default to all-1s
    # of the right length so a user toggling cache on via env still works.
    steps_mask = "1" * int(num_inference_steps) if enable_cache_dit else ""

    return FastFlux2Config(
        attention_backend=attention_backend,
        width=int(width),
        height=int(height),
        input_resize_mode="equivalent_area",
        num_inference_steps=int(num_inference_steps),
        guidance_scale=1.0,
        seed=0,
        enable_cache_dit=enable_cache_dit,
        cache_fn=1,
        cache_bn=0,
        residual_diff_threshold=0.8,
        steps_mask=steps_mask,
        steps_computation_policy="dynamic",
        enable_taylorseer=enable_taylorseer,
        taylorseer_order=1,
        compile_transformer=compile_transformer,
        compile_disable_cudagraphs=True,
        cache_timesteps=cache_timesteps,
        cache_image_latent_ids=cache_image_latent_ids,
        enable_vae_encoder_compile=enable_vae_encoder_compile,
        vae_encoder_compile_mode=vae_encoder_compile_mode,
        vae_encoder_compile_disable_cudagraphs=vae_encoder_compile_disable_cudagraphs,
        enable_vae_decoder_compile=enable_vae_decoder_compile,
        vae_decoder_compile_mode=vae_decoder_compile_mode,
        vae_decoder_compile_disable_cudagraphs=vae_decoder_compile_disable_cudagraphs,
        vae_decoder_channels_last=vae_decoder_channels_last,
        vae_decoder_input_channels_last=vae_decoder_input_channels_last,
        enable_taef2=enable_taef2,
        taef2_cache_dir=taef2_cache_dir,
        taef2_taesd_py_path=taef2_taesd_py_path,
        taef2_weight_path=taef2_weight_path,
        taef2_force_eager_vae=taef2_force_eager_vae,
        profile_stage_timing=profile_stage_timing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime img2img demo for FLUX.2 with fast config.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--attention-backend",
        choices=ATTENTION_BACKEND_CHOICES,
        default="auto",
    )
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be >= 1")

    api = RealtimeImg2ImgApi(
        build_default_config(
            attention_backend=args.attention_backend,
            num_inference_steps=args.num_inference_steps,
            width=args.width,
            height=args.height,
        )
    )

    uvicorn.run(
        api.app,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

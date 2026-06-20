from __future__ import annotations

import argparse
import asyncio
import base64
import os
import secrets
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .editor import FastFlux2Config, FastFlux2RealtimeEditor, normalize_attention_backend_name


DEFAULT_PROMPT = "A cinematic photo of a corgi astronaut walking on Mars at golden hour."
ATTENTION_BACKEND_CHOICES = ["auto", "sage", "native", "none", "sage_hub", "_flash_3", "fa3"]
STATIC_DIR = Path(__file__).resolve().parent / "static" / "realtime_txt2img"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class LoadInputModel(BaseModel):
    attention_backend: str = Field(default="auto")


class LoadResponseModel(BaseModel):
    status: str
    attention_backend: str


class PredictInputModel(BaseModel):
    prompt: str = Field(default=DEFAULT_PROMPT)
    seed: int = Field(default=-1, description="Use -1 for random seed")


class PredictResponseModel(BaseModel):
    base64_image: str
    seed: int
    request_tag: str
    total_ms: float
    refresh_ms: float
    prepare_ms: float
    decode_ms: float


class HealthResponseModel(BaseModel):
    status: str
    model_loaded: bool


class RealtimeTxt2ImgApi:
    def __init__(self, config: FastFlux2Config) -> None:
        self.editor = FastFlux2RealtimeEditor(config)
        self.app = FastAPI(title="Flux2 Realtime Txt2Img")

        self._model_lock = asyncio.Lock()

        self._setup_routes()

    @staticmethod
    def _normalize_attention_backend(attention_backend: str) -> str:
        return normalize_attention_backend_name(attention_backend)

    @staticmethod
    def _pil_to_base64(image, format: str = "JPEG") -> str:
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("ascii")

    @staticmethod
    def _resolve_seed(seed: int) -> int:
        if int(seed) >= 0:
            return int(seed)
        return secrets.randbelow(2**31 - 1)

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

        @self.app.post("/api/load", response_model=LoadResponseModel)
        async def load_model(inp: LoadInputModel) -> LoadResponseModel:
            async with self._model_lock:
                selected_backend = self._normalize_attention_backend(inp.attention_backend)
                if selected_backend not in ATTENTION_BACKEND_CHOICES:
                    selected_backend = "sage"

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

            async with self._model_lock:
                image, meta = self.editor.generate_text_to_image_with_meta(prompt=prompt, seed=seed)

            return PredictResponseModel(
                base64_image=self._pil_to_base64(image),
                seed=seed,
                request_tag=meta["request_tag"],
                total_ms=float(meta["total_ms"]),
                refresh_ms=float(meta["refresh_ms"]),
                prepare_ms=float(meta["prepare_ms"]),
                decode_ms=float(meta["decode_ms"]),
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

    Cache/compile/TaylorSeer/TAEF2 are OFF by default for max fidelity. Flip
    them back on via env vars if you want to trade quality for throughput.
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
    parser = argparse.ArgumentParser(description="Txt2img demo for FLUX.2 (HD, non-realtime).")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
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

    api = RealtimeTxt2ImgApi(
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

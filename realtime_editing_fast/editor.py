from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Optional

import numpy as np
from PIL import Image, ImageOps
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps

ATTENTION_BACKEND_ALIASES = {
    "fa3": "_flash_3",
    "flash3": "_flash_3",
    "flash_attn_3": "_flash_3",
    "flash-attn-3": "_flash_3",
    "flash_attention_3": "_flash_3",
    "default": "auto",
}
FA3_BACKENDS = {"_flash_3", "_flash_varlen_3"}
AUTO_ATTENTION_BACKENDS = {"auto"}


def normalize_attention_backend_name(name: str) -> str:
    value = (name or "").strip().lower()
    if not value:
        return "auto"
    return ATTENTION_BACKEND_ALIASES.get(value, value)


def _parse_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _parse_steps_mask(mask_text: str, expected_steps: int) -> list[int]:
    cleaned = mask_text.replace(",", "").replace(" ", "")
    if not cleaned:
        raise ValueError("steps mask cannot be empty")
    if any(ch not in ("0", "1") for ch in cleaned):
        raise ValueError(f"steps mask must only contain 0/1, got: {mask_text}")
    mask = [int(ch) for ch in cleaned]
    if len(mask) != expected_steps:
        raise ValueError(
            f"steps mask length mismatch: got {len(mask)}, expected {expected_steps} (num_inference_steps)",
        )
    return mask


def _parse_resample_mode(mode: str) -> Image.Resampling:
    value = (mode or "").strip().lower()
    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "box": Image.Resampling.BOX,
        "bilinear": Image.Resampling.BILINEAR,
        "hamming": Image.Resampling.HAMMING,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported preprocess resample mode: {mode}")
    return mapping[value]


@dataclass(slots=True)
class FastFlux2Config:
    model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    device: str = "cuda"
    gpu_id: int = 0
    dtype: str = "bfloat16"
    attention_backend: str = "auto"
    # Defaults target HD (1280x720) single-shot quality, not realtime throughput.
    width: int = 1280
    height: int = 720
    num_inference_steps: int = 28
    guidance_scale: float = 1.0
    seed: int = 0

    # Non-realtime defaults: keep cache/compile/taylorseer OFF for max fidelity
    # and to avoid the realtime-specific steps_mask length constraint.
    enable_cache_dit: bool = False
    cache_fn: int = 1
    cache_bn: int = 0
    residual_diff_threshold: float = 0.8
    single_block_rdt_scale: float = 3.0
    cache_max_warmup_steps: int = 0
    cache_warmup_interval: int = 1
    cache_max_cached_steps: int = -1
    cache_max_continuous_cached_steps: int = -1
    cache_enable_separate_cfg: bool = False
    # steps_mask is only consumed when enable_cache_dit=True. Leave empty here so
    # users who flip cache back on must explicitly provide a mask of the right length.
    steps_mask: str = ""
    steps_computation_policy: str = "dynamic"
    enable_taylorseer: bool = False
    taylorseer_order: int = 1

    compile_transformer: bool = False
    compile_disable_cudagraphs: bool = True
    # "equivalent_area" keeps aspect ratio and rescales source to ~width*height pixels.
    input_resize_mode: str = "equivalent_area"
    preprocess_fast_tensor: bool = True
    preprocess_resample: str = "bilinear"
    preprocess_pin_memory: bool = True
    preprocess_non_blocking_h2d: bool = True
    cache_timesteps: bool = False
    cache_image_latent_ids: bool = False
    enable_vae_encoder_compile: bool = False
    vae_encoder_compile_mode: str = "reduce-overhead"
    vae_encoder_compile_disable_cudagraphs: bool = True
    enable_vae_decoder_compile: bool = False
    vae_decoder_compile_mode: str = "reduce-overhead"
    vae_decoder_compile_disable_cudagraphs: bool = True
    vae_decoder_channels_last: bool = False
    vae_decoder_input_channels_last: bool = False
    # TAEF2 is a tiny VAE optimized for realtime; full VAE gives better HD quality.
    enable_taef2: bool = False
    taef2_cache_dir: str = ".cache/taef2"
    taef2_taesd_py_path: str = ""
    taef2_weight_path: str = ""
    taef2_force_eager_vae: bool = True
    profile_stage_timing: bool = False
    verbose: bool = True

    @property
    def runtime_device(self) -> str:
        if self.device == "cuda":
            return f"cuda:{self.gpu_id}"
        return self.device


class FastFlux2RealtimeEditor:
    def __init__(self, config: Optional[FastFlux2Config] = None):
        self.config = config or FastFlux2Config()
        self._pipe = None
        self._cache_dit_mod = None

        self._init_lock = threading.Lock()
        self._edit_lock = threading.Lock()

        self._cached_prompt: Optional[str] = None
        self._cached_prompt_embeds: Optional[torch.Tensor] = None
        self._cached_text_ids: Optional[torch.Tensor] = None
        self._request_idx = 0
        self._vae_encode_fn = None
        self._vae_decode_fn = None
        self._timesteps_cache: dict[tuple[int, int, str], torch.Tensor] = {}
        self._image_latent_ids_cache: dict[tuple[int, int, str], torch.Tensor] = {}
        self._fa3_probe_result: tuple[bool, str] | None = None
        self._fa3_dispatch_wrapper_ready: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def _log(self, msg: str) -> None:
        if not self.config.verbose:
            return
        ts = time.strftime("%H:%M:%S")
        print(f"*** [FastFlux2][{ts}] {msg}", flush=True)

    def _sync_if_cuda(self, force: bool = False) -> None:
        if not (force or self.config.profile_stage_timing):
            return
        if self.config.runtime_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _new_request_tag(self) -> str:
        self._request_idx += 1
        return f"req-{self._request_idx:06d}"

    @staticmethod
    def _round_to_multiple(v: float, multiple: int) -> int:
        return max(multiple, int(round(v / multiple) * multiple))

    def _compute_equivalent_resolution(
        self,
        src_w: int,
        src_h: int,
        target_area: int,
        multiple: int,
    ) -> tuple[int, int]:
        src_area = max(1.0, float(src_w * src_h))
        scale = (float(target_area) / src_area) ** 0.5
        proc_w = self._round_to_multiple(src_w * scale, multiple)
        proc_h = self._round_to_multiple(src_h * scale, multiple)
        return proc_w, proc_h

    def _probe_fa3_interface_compatibility(
        self,
        runtime_device: str,
        dtype: torch.dtype,
    ) -> tuple[bool, str]:
        if self._fa3_probe_result is not None:
            return self._fa3_probe_result

        if not runtime_device.startswith("cuda"):
            self._fa3_probe_result = (False, f"runtime_device={runtime_device} is not CUDA.")
            return self._fa3_probe_result

        if not torch.cuda.is_available():
            self._fa3_probe_result = (False, "torch.cuda.is_available() is False.")
            return self._fa3_probe_result

        if not self._fa3_dispatch_wrapper_ready:
            self._fa3_probe_result = (False, "FA3 dispatch wrapper is not ready.")
            return self._fa3_probe_result

        try:
            from diffusers.models import attention_dispatch as ad
        except Exception as exc:
            self._fa3_probe_result = (False, f"import diffusers attention_dispatch failed: {exc!r}")
            return self._fa3_probe_result

        fa3_func = getattr(ad, "flash_attn_3_func", None)
        if fa3_func is None:
            self._fa3_probe_result = (False, "attention_dispatch.flash_attn_3_func is None.")
            return self._fa3_probe_result

        probe_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        try:
            q = torch.randn((1, 1, 1, 64), device=runtime_device, dtype=probe_dtype)
            out = fa3_func(q=q, k=q, v=q, causal=False)
            if isinstance(out, tuple) and len(out) >= 2:
                self._fa3_probe_result = (True, "ok")
                return self._fa3_probe_result
            self._fa3_probe_result = (
                False,
                "patched attention_dispatch.flash_attn_3_func return is not tuple(out, lse, ...).",
            )
            return self._fa3_probe_result
        except Exception as exc:
            self._fa3_probe_result = (False, f"FA3 probe call failed: {exc!r}")
            return self._fa3_probe_result

    def _maybe_patch_diffusers_fa3_dispatch(self) -> tuple[bool, str]:
        if self._fa3_dispatch_wrapper_ready:
            return True, "already_patched"

        try:
            from diffusers.models import attention_dispatch as ad
        except Exception as exc:
            return False, f"import diffusers attention_dispatch failed: {exc!r}"

        if getattr(ad, "_flux_fa3_wrapper_installed", False):
            self._fa3_dispatch_wrapper_ready = True
            return True, "already_patched_global"

        base_fa3_func = getattr(ad, "flash_attn_3_func", None)
        base_fa3_varlen_func = getattr(ad, "flash_attn_3_varlen_func", None)
        if base_fa3_func is None or base_fa3_varlen_func is None:
            return False, "diffusers attention_dispatch missing FA3 function hooks."

        def _ensure_tuple_output(base_fn):
            def wrapped(*args, **kwargs):
                patched_kwargs = dict(kwargs)
                patched_kwargs.setdefault("return_attn_probs", True)
                out = base_fn(*args, **patched_kwargs)
                if isinstance(out, tuple):
                    return out
                if isinstance(out, torch.Tensor):
                    lse = torch.empty((0,), device=out.device, dtype=torch.float32)
                    return (out, lse)
                return out

            return wrapped

        ad.flash_attn_3_func = _ensure_tuple_output(base_fa3_func)
        ad.flash_attn_3_varlen_func = _ensure_tuple_output(base_fa3_varlen_func)
        ad._flux_fa3_wrapper_installed = True
        self._fa3_dispatch_wrapper_ready = True
        return True, "patched"

    def _try_set_attention_backend(self, pipe, backend: str) -> tuple[bool, str]:
        if not hasattr(pipe.transformer, "set_attention_backend"):
            return False, "Current transformer does not support set_attention_backend()."
        try:
            pipe.transformer.set_attention_backend(backend)
            return True, "ok"
        except Exception as exc:
            return False, repr(exc)

    def _auto_select_attention_backend(
        self,
        pipe,
        runtime_device: str,
        dtype: torch.dtype,
    ) -> str:
        for candidate in ("_flash_3", "sage", "native"):
            if candidate in FA3_BACKENDS:
                wrapped_ok, wrapped_reason = self._maybe_patch_diffusers_fa3_dispatch()
                if not wrapped_ok:
                    self._log(f"attention backend candidate skipped: {candidate}, reason={wrapped_reason}")
                    continue

                fa3_ok, fa3_reason = self._probe_fa3_interface_compatibility(
                    runtime_device=runtime_device,
                    dtype=dtype,
                )
                if not fa3_ok:
                    self._log(f"attention backend candidate skipped: {candidate}, reason={fa3_reason}")
                    continue

            ok, reason = self._try_set_attention_backend(pipe, candidate)
            if ok:
                self._log(f"attention backend auto-selected: {candidate}")
                return candidate
            self._log(f"attention backend candidate skipped: {candidate}, reason={reason}")

        self._log("attention backend auto-selection failed; using model default/native behavior.")
        return "native"

    def _maybe_enable_taef2_vae(
        self,
        pipe,
        runtime_device: str,
        dtype: torch.dtype,
    ) -> None:
        cfg = self.config
        if not cfg.enable_taef2:
            return

        try:
            from .taef2 import build_taef2_diffusers_vae, ensure_taef2_artifacts
        except Exception as exc:
            raise RuntimeError("TAEF2 requested but helper module import failed.") from exc

        taesd_py_path_cfg = (cfg.taef2_taesd_py_path or "").strip() or None
        taef2_weight_path_cfg = (cfg.taef2_weight_path or "").strip() or None
        cache_dir = Path(cfg.taef2_cache_dir or ".cache/taef2")
        taesd_py_path, taef2_weight_path = ensure_taef2_artifacts(
            cache_dir=cache_dir,
            taesd_py_path=taesd_py_path_cfg,
            taef2_weight_path=taef2_weight_path_cfg,
        )

        bn_channels = 128
        batch_norm_eps = 0.0
        if hasattr(pipe.vae, "bn") and hasattr(pipe.vae.bn, "running_mean"):
            bn_channels = int(pipe.vae.bn.running_mean.shape[0])
            batch_norm_eps = float(getattr(pipe.vae.bn, "eps", 0.0))
        elif hasattr(pipe.vae, "config") and hasattr(pipe.vae.config, "batch_norm_eps"):
            batch_norm_eps = float(pipe.vae.config.batch_norm_eps)

        pipe.vae = build_taef2_diffusers_vae(
            taesd_py_path=taesd_py_path,
            taef2_weight_path=taef2_weight_path,
            device=runtime_device,
            dtype=dtype,
            bn_channels=bn_channels,
            batch_norm_eps=batch_norm_eps,
        )
        self._image_latent_ids_cache.clear()
        self._log(
            "TAEF2 enabled: "
            f"weights={taef2_weight_path}, taesd={taesd_py_path}, force_eager_vae={cfg.taef2_force_eager_vae}"
        )

        if cfg.taef2_force_eager_vae:
            cfg.enable_vae_encoder_compile = False
            cfg.enable_vae_decoder_compile = False
            self._log("TAEF2 forcing VAE eager paths (encoder/decoder compile disabled).")

    def _configure_vae_runtime(self, pipe) -> None:
        cfg = self.config

        if cfg.vae_decoder_channels_last and hasattr(pipe.vae, "decoder"):
            pipe.vae.decoder.to(memory_format=torch.channels_last)
            self._log("vae decoder memory_format=channels_last")

        def _encode_fn(image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
            return pipe._encode_vae_image(image=image, generator=generator)

        if cfg.enable_vae_encoder_compile:
            if cfg.vae_encoder_compile_disable_cudagraphs:
                encode_compile_kwargs = {"fullgraph": False, "options": {"triton.cudagraphs": False}}
            else:
                encode_compile_kwargs = {"mode": cfg.vae_encoder_compile_mode, "fullgraph": False}
            try:
                self._vae_encode_fn = torch.compile(_encode_fn, **encode_compile_kwargs)
                self._log(f"vae encoder compiled: kwargs={encode_compile_kwargs}")
            except Exception as exc:
                self._vae_encode_fn = _encode_fn
                self._log(f"vae encoder compile fallback to eager: {exc!r}")
        else:
            self._vae_encode_fn = _encode_fn

        def _decode_fn(latents: torch.Tensor) -> torch.Tensor:
            return pipe.vae.decode(latents, return_dict=False)[0]

        if cfg.enable_vae_decoder_compile:
            if cfg.vae_decoder_compile_disable_cudagraphs:
                decode_compile_kwargs = {"fullgraph": False, "options": {"triton.cudagraphs": False}}
            else:
                decode_compile_kwargs = {"mode": cfg.vae_decoder_compile_mode, "fullgraph": False}
            try:
                self._vae_decode_fn = torch.compile(_decode_fn, **decode_compile_kwargs)
                self._log(f"vae decoder compiled: kwargs={decode_compile_kwargs}")
            except Exception as exc:
                self._vae_decode_fn = _decode_fn
                self._log(f"vae decoder compile fallback to eager: {exc!r}")
        else:
            self._vae_decode_fn = _decode_fn

    def ensure_loaded(self) -> None:
        if self._pipe is not None:
            return

        with self._init_lock:
            if self._pipe is not None:
                return

            cfg = self.config
            runtime_device = cfg.runtime_device
            dtype = _parse_dtype(cfg.dtype)
            self._log(f"model init start: model={cfg.model_id}, device={runtime_device}, dtype={dtype}")
            init_t0 = time.perf_counter()

            if runtime_device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
                torch.cuda.set_device(runtime_device)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            pipe = AutoPipelineForImage2Image.from_pretrained(cfg.model_id, torch_dtype=dtype)
            pipe = pipe.to(runtime_device)
            pipe.set_progress_bar_config(disable=True)
            requested_attention_backend = normalize_attention_backend_name(cfg.attention_backend)
            if requested_attention_backend in AUTO_ATTENTION_BACKENDS:
                selected_backend = self._auto_select_attention_backend(
                    pipe=pipe,
                    runtime_device=runtime_device,
                    dtype=dtype,
                )
                cfg.attention_backend = selected_backend
            elif requested_attention_backend not in ("", "none"):
                if requested_attention_backend in FA3_BACKENDS:
                    wrapped_ok, wrapped_reason = self._maybe_patch_diffusers_fa3_dispatch()
                    if wrapped_ok:
                        self._log(f"FA3 dispatch wrapper status: {wrapped_reason}")
                    else:
                        self._log(f"FA3 dispatch wrapper failed: {wrapped_reason}")

                    fa3_ok, fa3_reason = self._probe_fa3_interface_compatibility(
                        runtime_device=runtime_device,
                        dtype=dtype,
                    )
                    if not fa3_ok:
                        fallback_backend = "sage"
                        self._log(
                            "FA3 backend probe failed; fallback to "
                            f"{fallback_backend}. reason={fa3_reason}"
                        )
                        requested_attention_backend = fallback_backend

                if not hasattr(pipe.transformer, "set_attention_backend"):
                    raise RuntimeError("Current transformer does not support set_attention_backend().")
                try:
                    pipe.transformer.set_attention_backend(requested_attention_backend)
                    cfg.attention_backend = requested_attention_backend
                    self._log(f"attention backend set: {requested_attention_backend}")
                except Exception as exc:
                    if requested_attention_backend in FA3_BACKENDS:
                        for fallback_backend in ("sage", "native"):
                            if fallback_backend == requested_attention_backend:
                                continue
                            try:
                                pipe.transformer.set_attention_backend(fallback_backend)
                                cfg.attention_backend = fallback_backend
                                self._log(
                                    "attention backend fallback set: "
                                    f"{requested_attention_backend} -> {fallback_backend}"
                                )
                                break
                            except Exception:
                                continue
                        else:
                            raise RuntimeError(
                                f"Failed to set attention backend to '{requested_attention_backend}'."
                            ) from exc
                        self._log(f"attention backend set failed for FA3: {exc!r}")
                    else:
                        raise RuntimeError(
                            f"Failed to set attention backend to '{requested_attention_backend}'."
                        ) from exc
            else:
                cfg.attention_backend = requested_attention_backend

            self._maybe_enable_taef2_vae(
                pipe=pipe,
                runtime_device=runtime_device,
                dtype=dtype,
            )

            if cfg.enable_cache_dit:
                self._enable_cache_dit(pipe)

            if cfg.compile_transformer:
                if self._cache_dit_mod is not None:
                    self._cache_dit_mod.set_compile_configs()

                if cfg.compile_disable_cudagraphs:
                    compile_kwargs = {"fullgraph": False, "options": {"triton.cudagraphs": False}}
                else:
                    compile_kwargs = {"mode": "reduce-overhead", "fullgraph": False}

                pipe.transformer = torch.compile(pipe.transformer, **compile_kwargs)
                self._log(f"transformer compiled: kwargs={compile_kwargs}")

            self._configure_vae_runtime(pipe)

            self._pipe = pipe
            self._sync_if_cuda(force=True)
            self._log(f"model init done: {(time.perf_counter() - init_t0) * 1000.0:.1f} ms")

    def _enable_cache_dit(self, pipe) -> None:
        cfg = self.config
        try:
            import cache_dit as cache_dit_mod
            from cache_dit import (
                BlockAdapter,
                DBCacheConfig,
                ForwardPattern,
                ParamsModifier,
                TaylorSeerCalibratorConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "cache-dit is not available. Install it first, e.g. `uv pip install -U cache-dit`."
            ) from exc

        steps_mask = _parse_steps_mask(cfg.steps_mask, cfg.num_inference_steps)

        cache_config = DBCacheConfig(
            Fn_compute_blocks=cfg.cache_fn,
            Bn_compute_blocks=cfg.cache_bn,
            residual_diff_threshold=cfg.residual_diff_threshold,
            max_warmup_steps=cfg.cache_max_warmup_steps,
            warmup_interval=cfg.cache_warmup_interval,
            max_cached_steps=cfg.cache_max_cached_steps,
            max_continuous_cached_steps=cfg.cache_max_continuous_cached_steps,
            enable_separate_cfg=cfg.cache_enable_separate_cfg,
            num_inference_steps=cfg.num_inference_steps,
            steps_computation_mask=steps_mask,
            steps_computation_policy=cfg.steps_computation_policy,
        )

        cache_config_cls = cache_config.__class__
        params_modifiers = [
            ParamsModifier(
                cache_config=cache_config_cls().reset(
                    residual_diff_threshold=cfg.residual_diff_threshold,
                ),
            ),
            ParamsModifier(
                cache_config=cache_config_cls().reset(
                    residual_diff_threshold=cfg.residual_diff_threshold * cfg.single_block_rdt_scale,
                ),
            ),
        ]

        calibrator_config = (
            TaylorSeerCalibratorConfig(taylorseer_order=cfg.taylorseer_order)
            if cfg.enable_taylorseer
            else None
        )

        # Use transformer-only adapter because this realtime path bypasses pipe.__call__.
        cache_adapter = BlockAdapter(
            pipe=None,
            transformer=pipe.transformer,
            blocks=[
                pipe.transformer.transformer_blocks,
                pipe.transformer.single_transformer_blocks,
            ],
            forward_pattern=[
                ForwardPattern.Pattern_1,
                ForwardPattern.Pattern_3,
            ],
        )

        cache_dit_mod.enable_cache(
            cache_adapter,
            cache_config=cache_config,
            calibrator_config=calibrator_config,
            params_modifiers=params_modifiers,
        )
        self._cache_dit_mod = cache_dit_mod

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        pipe = self._pipe
        assert pipe is not None

        if (
            prompt == self._cached_prompt
            and self._cached_prompt_embeds is not None
            and self._cached_text_ids is not None
        ):
            return self._cached_prompt_embeds, self._cached_text_ids

        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_embeds=None,
            device=pipe._execution_device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )

        self._cached_prompt = prompt
        self._cached_prompt_embeds = prompt_embeds
        self._cached_text_ids = text_ids
        return prompt_embeds, text_ids

    def _decode_latents_to_pil(
        self,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
    ) -> Image.Image:
        pipe = self._pipe
        assert pipe is not None

        latents = pipe._unpack_latents_with_ids(latents, latent_ids)
        latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(
            latents.device,
            latents.dtype,
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = pipe._unpatchify_latents(latents)
        if self.config.vae_decoder_input_channels_last:
            latents = latents.contiguous(memory_format=torch.channels_last)

        if self._vae_decode_fn is None:
            image = pipe.vae.decode(latents, return_dict=False)[0]
        else:
            try:
                image = self._vae_decode_fn(latents)
            except Exception as exc:
                self._log(f"vae decoder compiled path failed, fallback to eager: {exc!r}")
                self._vae_decode_fn = None
                image = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")[0]

    def _preprocess_image_tensor(
        self,
        image_for_preprocess: Image.Image,
        target_h: int,
        target_w: int,
        resize_mode: str,
    ) -> torch.Tensor:
        pipe = self._pipe
        cfg = self.config
        assert pipe is not None

        if not cfg.preprocess_fast_tensor:
            return pipe.image_processor.preprocess(
                image_for_preprocess,
                height=target_h,
                width=target_w,
                resize_mode=resize_mode,
            )

        if image_for_preprocess.size != (target_w, target_h):
            image_for_preprocess = image_for_preprocess.resize(
                (target_w, target_h),
                _parse_resample_mode(cfg.preprocess_resample),
            )

        arr = np.array(image_for_preprocess, dtype=np.uint8, copy=True)
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(torch.float32).div_(127.5).sub_(1.0)

        if cfg.preprocess_pin_memory and torch.cuda.is_available():
            image_tensor = image_tensor.pin_memory()
        return image_tensor

    def _prepare_image_latents_fast(
        self,
        image_tensor: torch.Tensor,
        generator: torch.Generator,
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pipe = self._pipe
        cfg = self.config
        assert pipe is not None

        non_blocking = bool(cfg.preprocess_non_blocking_h2d)
        image = image_tensor.to(
            device=pipe._execution_device,
            dtype=pipe.vae.dtype,
            non_blocking=non_blocking,
        )
        if self._vae_encode_fn is None:
            image_latent = pipe._encode_vae_image(image=image, generator=generator)
        else:
            try:
                image_latent = self._vae_encode_fn(image=image, generator=generator)
            except Exception as exc:
                self._log(f"vae encoder compiled path failed, fallback to eager: {exc!r}")
                self._vae_encode_fn = None
                image_latent = pipe._encode_vae_image(image=image, generator=generator)
        packed_latent = pipe._pack_latents(image_latent)
        if batch_size != 1:
            packed_latent = packed_latent.repeat(batch_size, 1, 1)

        image_latent_h = int(image_latent.shape[-2])
        image_latent_w = int(image_latent.shape[-1])
        latent_ids_cache_key = (image_latent_h, image_latent_w, str(pipe._execution_device))
        if cfg.cache_image_latent_ids and latent_ids_cache_key in self._image_latent_ids_cache:
            image_latent_ids = self._image_latent_ids_cache[latent_ids_cache_key]
        else:
            image_latent_ids = pipe._prepare_image_ids([image_latent]).to(
                pipe._execution_device,
                non_blocking=non_blocking,
            )
            if cfg.cache_image_latent_ids:
                self._image_latent_ids_cache[latent_ids_cache_key] = image_latent_ids

        if batch_size != 1:
            image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
        return packed_latent, image_latent_ids

    def _get_timesteps_for_latent_seq_len(self, image_seq_len: int) -> torch.Tensor:
        pipe = self._pipe
        cfg = self.config
        assert pipe is not None

        cache_key = (int(image_seq_len), int(cfg.num_inference_steps), str(pipe._execution_device))
        if cfg.cache_timesteps and cache_key in self._timesteps_cache:
            return self._timesteps_cache[cache_key]

        sigmas = np.linspace(1.0, 1.0 / cfg.num_inference_steps, cfg.num_inference_steps)
        if hasattr(pipe.scheduler.config, "use_flow_sigmas") and pipe.scheduler.config.use_flow_sigmas:
            sigmas = None

        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=cfg.num_inference_steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            cfg.num_inference_steps,
            pipe._execution_device,
            sigmas=sigmas,
            mu=mu,
        )
        if cfg.cache_timesteps:
            self._timesteps_cache[cache_key] = timesteps
        return timesteps

    def _prepare_inputs(
        self,
        image: Image.Image,
        prompt: str,
        seed: int,
    ) -> dict:
        pipe = self._pipe
        cfg = self.config
        assert pipe is not None

        prompt_embeds, text_ids = self._encode_prompt(prompt)

        image = image.convert("RGB")
        source_size = image.size
        multiple = int(pipe.vae_scale_factor * 2)
        resample = _parse_resample_mode(cfg.preprocess_resample)
        target_area = int(cfg.width * cfg.height)
        if cfg.input_resize_mode == "equivalent_area":
            proc_w, proc_h = self._compute_equivalent_resolution(
                src_w=source_size[0],
                src_h=source_size[1],
                target_area=target_area,
                multiple=multiple,
            )
            image_for_preprocess = image.resize((proc_w, proc_h), resample)
            resize_mode = "crop"  # no crop happens because image already matches target size.
            target_w, target_h = proc_w, proc_h
        elif cfg.input_resize_mode == "crop":
            image_for_preprocess = image
            resize_mode = "crop"
            target_w, target_h = cfg.width, cfg.height
        elif cfg.input_resize_mode == "pad":
            contained = ImageOps.contain(image, (cfg.width, cfg.height), method=resample)
            canvas = Image.new("RGB", (cfg.width, cfg.height), (0, 0, 0))
            offset = ((cfg.width - contained.width) // 2, (cfg.height - contained.height) // 2)
            canvas.paste(contained, offset)
            image_for_preprocess = canvas
            resize_mode = "crop"
            target_w, target_h = cfg.width, cfg.height
        else:
            raise ValueError(f"Unsupported input_resize_mode: {cfg.input_resize_mode}")

        image_tensor = self._preprocess_image_tensor(
            image_for_preprocess,
            target_h=target_h,
            target_w=target_w,
            resize_mode=resize_mode,
        )

        generator = torch.Generator(device=cfg.runtime_device).manual_seed(seed)
        num_channels_latents = pipe.transformer.config.in_channels // 4

        latents, latent_ids = pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=num_channels_latents,
            height=target_h,
            width=target_w,
            dtype=prompt_embeds.dtype,
            device=pipe._execution_device,
            generator=generator,
            latents=None,
        )

        image_latents, image_latent_ids = self._prepare_image_latents_fast(
            image_tensor=image_tensor,
            generator=generator,
            batch_size=1,
        )

        image_seq_len = latents.shape[1]
        timesteps = self._get_timesteps_for_latent_seq_len(image_seq_len=image_seq_len)

        return {
            "latents": latents,
            "latent_ids": latent_ids,
            "image_latents": image_latents,
            "image_latent_ids": image_latent_ids,
            "timesteps": timesteps,
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
            "source_size": source_size,
            "target_size": (target_w, target_h),
        }

    def _prepare_t2i_inputs(
        self,
        prompt: str,
        seed: int,
    ) -> dict:
        pipe = self._pipe
        cfg = self.config
        assert pipe is not None

        prompt_embeds, text_ids = self._encode_prompt(prompt)
        target_h = int(cfg.height)
        target_w = int(cfg.width)

        generator = torch.Generator(device=cfg.runtime_device).manual_seed(seed)
        num_channels_latents = pipe.transformer.config.in_channels // 4

        latents, latent_ids = pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=num_channels_latents,
            height=target_h,
            width=target_w,
            dtype=prompt_embeds.dtype,
            device=pipe._execution_device,
            generator=generator,
            latents=None,
        )

        image_seq_len = latents.shape[1]
        timesteps = self._get_timesteps_for_latent_seq_len(image_seq_len=image_seq_len)

        return {
            "latents": latents,
            "latent_ids": latent_ids,
            "timesteps": timesteps,
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
            "target_size": (target_w, target_h),
        }

    @torch.no_grad()
    def edit_image_with_meta(
        self,
        image: Image.Image,
        prompt: str,
        seed: Optional[int] = None,
    ) -> tuple[Image.Image, dict]:
        if image is None:
            raise ValueError("Input image is required.")
        if not prompt:
            raise ValueError("Prompt is required.")

        self.ensure_loaded()
        pipe = self._pipe
        cfg = self.config
        assert pipe is not None

        seed = cfg.seed if seed is None else int(seed)
        req_tag = self._new_request_tag()
        total_t0 = time.perf_counter()
        self._log(f"{req_tag} begin: seed={seed}, prompt_len={len(prompt)}, resize_mode={cfg.input_resize_mode}")

        with self._edit_lock:
            stage_t0 = time.perf_counter()
            if self._cache_dit_mod is not None:
                self._cache_dit_mod.refresh_context(
                    pipe.transformer,
                    num_inference_steps=cfg.num_inference_steps,
                    verbose=False,
                )
            self._sync_if_cuda()
            refresh_ms = (time.perf_counter() - stage_t0) * 1000.0
            self._log(f"{req_tag} stage refresh_context: {refresh_ms:.1f} ms")

            stage_t0 = time.perf_counter()
            inputs = self._prepare_inputs(image=image, prompt=prompt, seed=seed)
            self._sync_if_cuda()
            prepare_ms = (time.perf_counter() - stage_t0) * 1000.0
            self._log(
                f"{req_tag} stage prepare_inputs: {prepare_ms:.1f} ms, source={inputs['source_size']}, "
                f"target={inputs['target_size']}, latents={tuple(inputs['latents'].shape)}"
            )
            latents = inputs["latents"]
            latent_ids = inputs["latent_ids"]
            image_latents = inputs["image_latents"]
            image_latent_ids = inputs["image_latent_ids"]

            pipe.scheduler.set_begin_index(0)
            pipe.scheduler._step_index = None

            step_ms: list[float] = []
            for timestep_value in inputs["timesteps"]:
                step_t0 = time.perf_counter()
                timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)

                latent_model_input = torch.cat([latents, image_latents], dim=1).to(pipe.transformer.dtype)
                latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()

                with pipe.transformer.cache_context("cond"):
                    noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=inputs["prompt_embeds"],
                        txt_ids=inputs["text_ids"],
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]

                noise_pred = noise_pred[:, : latents.size(1), :]
                latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
                self._sync_if_cuda()
                one_step_ms = (time.perf_counter() - step_t0) * 1000.0
                step_ms.append(one_step_ms)
                self._log(f"{req_tag} stage denoise_step[{len(step_ms)}]: {one_step_ms:.1f} ms")

            stage_t0 = time.perf_counter()
            decoded = self._decode_latents_to_pil(latents, latent_ids)
            if decoded.size != inputs["source_size"]:
                decoded = decoded.resize(inputs["source_size"], Image.Resampling.BICUBIC)
            self._sync_if_cuda()
            decode_ms = (time.perf_counter() - stage_t0) * 1000.0
            total_ms = (time.perf_counter() - total_t0) * 1000.0
            self._log(f"{req_tag} stage decode: {decode_ms:.1f} ms")
            self._log(f"{req_tag} done total: {total_ms:.1f} ms")

            meta = {
                "request_tag": req_tag,
                "refresh_ms": refresh_ms,
                "prepare_ms": prepare_ms,
                "step_ms": step_ms,
                "decode_ms": decode_ms,
                "total_ms": total_ms,
                "source_size": inputs["source_size"],
                "target_size": inputs["target_size"],
                "resize_mode": cfg.input_resize_mode,
            }
            return decoded, meta

    @torch.no_grad()
    def edit_image(self, image: Image.Image, prompt: str, seed: Optional[int] = None) -> Image.Image:
        edited, _ = self.edit_image_with_meta(image=image, prompt=prompt, seed=seed)
        return edited

    @torch.no_grad()
    def generate_text_to_image_with_meta(
        self,
        prompt: str,
        seed: Optional[int] = None,
    ) -> tuple[Image.Image, dict]:
        if not prompt:
            raise ValueError("Prompt is required.")

        self.ensure_loaded()
        pipe = self._pipe
        cfg = self.config
        assert pipe is not None

        seed = cfg.seed if seed is None else int(seed)
        req_tag = self._new_request_tag()
        total_t0 = time.perf_counter()
        self._log(f"{req_tag} begin txt2img: seed={seed}, prompt_len={len(prompt)}")

        with self._edit_lock:
            stage_t0 = time.perf_counter()
            if self._cache_dit_mod is not None:
                self._cache_dit_mod.refresh_context(
                    pipe.transformer,
                    num_inference_steps=cfg.num_inference_steps,
                    verbose=False,
                )
            self._sync_if_cuda()
            refresh_ms = (time.perf_counter() - stage_t0) * 1000.0
            self._log(f"{req_tag} stage refresh_context: {refresh_ms:.1f} ms")

            stage_t0 = time.perf_counter()
            inputs = self._prepare_t2i_inputs(prompt=prompt, seed=seed)
            self._sync_if_cuda()
            prepare_ms = (time.perf_counter() - stage_t0) * 1000.0
            self._log(
                f"{req_tag} stage prepare_t2i_inputs: {prepare_ms:.1f} ms, "
                f"target={inputs['target_size']}, latents={tuple(inputs['latents'].shape)}"
            )

            latents = inputs["latents"]
            latent_ids = inputs["latent_ids"]

            pipe.scheduler.set_begin_index(0)
            pipe.scheduler._step_index = None

            step_ms: list[float] = []
            for timestep_value in inputs["timesteps"]:
                step_t0 = time.perf_counter()
                timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
                latent_model_input = latents.to(pipe.transformer.dtype)

                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()

                with pipe.transformer.cache_context("cond"):
                    noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=inputs["prompt_embeds"],
                        txt_ids=inputs["text_ids"],
                        img_ids=latent_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]

                noise_pred = noise_pred[:, : latents.size(1), :]
                latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
                self._sync_if_cuda()
                one_step_ms = (time.perf_counter() - step_t0) * 1000.0
                step_ms.append(one_step_ms)
                self._log(f"{req_tag} stage denoise_step[{len(step_ms)}]: {one_step_ms:.1f} ms")

            stage_t0 = time.perf_counter()
            decoded = self._decode_latents_to_pil(latents, latent_ids)
            self._sync_if_cuda()
            decode_ms = (time.perf_counter() - stage_t0) * 1000.0
            total_ms = (time.perf_counter() - total_t0) * 1000.0
            self._log(f"{req_tag} stage decode: {decode_ms:.1f} ms")
            self._log(f"{req_tag} done txt2img total: {total_ms:.1f} ms")

            meta = {
                "request_tag": req_tag,
                "refresh_ms": refresh_ms,
                "prepare_ms": prepare_ms,
                "step_ms": step_ms,
                "decode_ms": decode_ms,
                "total_ms": total_ms,
                "source_size": None,
                "target_size": inputs["target_size"],
                "resize_mode": "txt2img",
            }
            return decoded, meta

    @torch.no_grad()
    def generate_text_to_image(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        image, _ = self.generate_text_to_image_with_meta(prompt=prompt, seed=seed)
        return image

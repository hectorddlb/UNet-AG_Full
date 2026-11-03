"""DRUS diffusion prior: model configuration and loading.

The method compared in the thesis uses the prior **without fine-tuning**: the
unconditional 256×256 guided-diffusion model trained on ImageNet and published
by OpenAI (``256x256_diffusion_uncond.pt``, 552.8 M parameters). It is the
best-performing variant of the replica.

``MODEL_FLAGS`` and ``DIFFUSION_FLAGS`` are those of the published checkpoint
(= ``configs/imagenet_256.yml`` of the DRUS-v1 repository); if they change, the
``state_dict`` no longer loads. This module is the single source of those
values: the fine-tuning, sampling and operation-counting scripts import them
from here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

__all__ = ["MODEL_FLAGS", "DIFFUSION_FLAGS", "PRIOR_FILENAME", "PRIOR_URL",
           "PRIOR_SIZE_BYTES", "PRIOR_MD5", "load_prior", "n_parameters"]

PRIOR_FILENAME = "256x256_diffusion_uncond.pt"
PRIOR_URL = ("https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
             + PRIOR_FILENAME)
PRIOR_SIZE_BYTES = 2_211_383_297
PRIOR_MD5 = "fd9dd2335b8736d521de0aed54bd90ca"

MODEL_FLAGS = dict(
    image_size=256,
    num_channels=256,
    num_res_blocks=2,
    channel_mult="",
    learn_sigma=True,
    class_cond=False,
    use_checkpoint=True,          # gradient checkpointing: needed on 24 GB
    attention_resolutions="32,16,8",
    num_heads=4,
    num_head_channels=64,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    dropout=0.0,
    resblock_updown=True,
    use_fp16=True,
    use_new_attention_order=False,
)
DIFFUSION_FLAGS = dict(
    diffusion_steps=1000,
    noise_schedule="linear",
    timestep_respacing="",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
)


def load_prior(checkpoint: Union[str, Path], device: str = "cuda",
               fp16: Union[bool, None] = None) -> torch.nn.Module:
    """Instantiate the guided-diffusion UNet and load the prior weights.

    Reproduces the loading with which the thesis results were generated
    (``methods/04_drus/scripts/sample_ddrm.py``): weights in ``float32`` and
    conversion of the convolutional blocks to ``float16``, in evaluation mode.
    ``fp16=None`` applies that conversion only on GPU; on CPU the model stays
    in ``float32`` (and the result is no longer identical to the thesis one).
    """
    from guided_diffusion.script_util import create_model_and_diffusion

    flags = dict(MODEL_FLAGS)
    if fp16 is None:
        fp16 = torch.device(device).type == "cuda"
    flags["use_fp16"] = bool(fp16)
    model, _ = create_model_and_diffusion(**flags, **DIFFUSION_FLAGS)
    state = torch.load(Path(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    if flags["use_fp16"]:
        model.convert_to_fp16()
    model.eval()
    return model


def n_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob

import torch


UNET_DECODER_PREFIXES = (
    "model.2.",
    "model.1.submodule.2.",
    "model.1.submodule.1.submodule.2.",
    "model.1.submodule.1.submodule.1.submodule.2.",
)

UNET_BOTTLENECK_PREFIX = "model.1.submodule.1.submodule.1.submodule.1.submodule."

UNET_FREEZE_MODES = {
    "none": None,
    "head": ("model.2.",),
    "decoder": UNET_DECODER_PREFIXES,
    "decoder_bottleneck": UNET_DECODER_PREFIXES + (UNET_BOTTLENECK_PREFIX,),
}

NNUNET_FREEZE_MODES = {
    "none": None,
    "head": ("decoder.seg_layers.",),
    "decoder": ("decoder.",),
    "decoder_bottleneck": ("decoder.", "encoder.stages.5."),
}

SSM_FREEZE_MODES = {
    "none": None,
    "head": ("head.",),
    "decoder": ("decoder.", "head."),
    "decoder_bottleneck": ("decoder.", "head.", "bottleneck."),
}

SWINUNETR_FREEZE_MODES = {
    "none": None,
    "head": ("out.",),
    "decoder": ("decoder", "out."),
    "decoder_bottleneck": ("decoder", "encoder10.", "out."),
}

FREEZE_MODES_BY_MODEL = {
    "UNet": UNET_FREEZE_MODES,
    "nnUNet": NNUNET_FREEZE_MODES,
    "SSM": SSM_FREEZE_MODES,
    "SwinUNETR": SWINUNETR_FREEZE_MODES,
}


def get_freeze_modes():
    modes = set()
    for model_modes in FREEZE_MODES_BY_MODEL.values():
        modes.update(model_modes.keys())
    return tuple(sorted(modes))


def resolve_pretrained_checkpoint(model_dir):
    model_dir = os.path.normpath(model_dir)

    if os.path.isfile(model_dir):
        return model_dir

    if not os.path.isdir(model_dir):
        raise ValueError(f"--pretrained må peke til en modellmappe eller checkpoint-fil, fikk: {model_dir}")

    for filename in (
        "best_model.pth",
        "model.pt",
        "model.pth",
        "checkpoint.pt",
        "checkpoint.pth",
    ):
        ckpt_path = os.path.join(model_dir, filename)
        if os.path.isfile(ckpt_path):
            return ckpt_path

    nested_candidates = []
    for filename in (
        "best_model.pth",
        "model.pt",
        "model.pth",
        "checkpoint.pt",
        "checkpoint.pth",
    ):
        nested_candidates.extend(
            glob.glob(os.path.join(model_dir, "**", filename), recursive=True)
        )

    nested_candidates = sorted(set(nested_candidates))
    if len(nested_candidates) == 1:
        return nested_candidates[0]

    if len(nested_candidates) > 1:
        raise FileNotFoundError(
            f"Fant flere mulige checkpoints under '{model_dir}'. Pek direkte på en av dem:\n"
            + "\n".join(f"  - {path}" for path in nested_candidates)
        )

    raise FileNotFoundError(
        f"Fant ikke checkpoint i '{model_dir}'. Forventer best_model.pth, model.pth, "
        "model.pt, checkpoint.pth eller checkpoint.pt."
    )


def build_finetuned_model_name(model_dir, datasets, source_model_name=None):
    dataset_suffix = "+".join(sorted(datasets))
    model_dir = os.path.normpath(model_dir)
    source_model_name = source_model_name or os.path.basename(model_dir)
    return f"{source_model_name}_finetuned_{dataset_suffix}"


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]

    for key in ("state_dict", "model_state_dict", "network_weights"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            return checkpoint[key]

    if isinstance(checkpoint, dict):
        return checkpoint

    raise ValueError("Unsupported checkpoint format")


def _normalize_state_dict_keys(state_dict, model):
    model_keys = set(model.state_dict().keys())
    normalized = {}
    for key, value in state_dict.items():
        candidates = [key]
        for prefix in ("module.", "network."):
            if key.startswith(prefix):
                candidates.append(key[len(prefix):])

        clean_key = next((candidate for candidate in candidates if candidate in model_keys), candidates[0])
        normalized[clean_key] = value
    return normalized


def _filter_shape_mismatches(model, state_dict):
    model_state = model.state_dict()
    compatible = {}
    skipped = []

    for key, value in state_dict.items():
        if key not in model_state:
            compatible[key] = value
            continue

        if tuple(value.shape) == tuple(model_state[key].shape):
            compatible[key] = value
        else:
            skipped.append((key, tuple(value.shape), tuple(model_state[key].shape)))

    return compatible, skipped


def load_pretrained_checkpoint(model, model_dir, device):
    ckpt_path = resolve_pretrained_checkpoint(model_dir)
    print(f"Laster pretrained checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = _normalize_state_dict_keys(_extract_state_dict(checkpoint), model)
    state_dict, skipped = _filter_shape_mismatches(model, state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("Modellvekter lastet")
    if skipped:
        print("  Hoppet over vekter med annen shape:")
        for name, checkpoint_shape, model_shape in skipped:
            print(f"    {name}: checkpoint {checkpoint_shape} -> modell {model_shape}")
    if missing:
        print("  Mangler:", missing)
    if unexpected:
        print("  Uventede:", unexpected)

    return missing, unexpected


def get_freeze_modes_for_model(model_name):
    if model_name not in FREEZE_MODES_BY_MODEL:
        raise ValueError(
            f"Ukjent model_name='{model_name}'. Gyldige: {list(FREEZE_MODES_BY_MODEL.keys())}"
        )
    return FREEZE_MODES_BY_MODEL[model_name]


def apply_freeze_mode(model, freeze_mode, model_name="UNet"):
    freeze_modes = get_freeze_modes_for_model(model_name)

    if freeze_mode not in freeze_modes:
        raise ValueError(
            f"Ukjent freeze_mode='{freeze_mode}' for model_name='{model_name}'. "
            f"Gyldige: {list(freeze_modes.keys())}"
        )

    prefixes = freeze_modes[freeze_mode]
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()

        if prefixes is None:
            should_train = True
        else:
            should_train = any(name.startswith(prefix) for prefix in prefixes)

        param.requires_grad = should_train

        if should_train:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()

    print(f"Freeze mode: {freeze_mode} ({model_name})")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Frozen params:    {frozen_params:,}")
    print(f"  Total params:     {total_params:,}")

    if trainable_params == 0:
        raise RuntimeError(
            f"freeze_mode='{freeze_mode}' for model_name='{model_name}' gjorde ingen "
            "parametre trainable. Sjekk prefixene i utils_finetuning.py."
        )

    return {
        "model_name": model_name,
        "freeze_mode": freeze_mode,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_params": total_params,
    }


def prepare_model_for_finetuning(model, model_dir, device, freeze_mode, model_name="UNet"):
    load_pretrained_checkpoint(model, model_dir, device)
    return apply_freeze_mode(model, freeze_mode, model_name=model_name)

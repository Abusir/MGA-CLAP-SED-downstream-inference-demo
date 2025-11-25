#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal example for frame-level text–audio prediction with ASE / MGA-CLAP,
and visualization as a time–class probability heatmap.

This script fixes several missing pieces in the original repo so that you can
easily reproduce “per-frame class probability over time”.
"""

from typing import List, Tuple
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from ruamel import yaml
import matplotlib.pyplot as plt

from models.ase_model import ASE


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(cfg_path: str, text_classes: List[str], audio_path: str) -> None:
    """Run ASE on one audio clip and plot frame-level predictions."""
    model, device = load_model(cfg_path)

    waveform, audio_len = load_audio(audio_path, device)
    model.eval()

    with torch.no_grad():
        # ----- text encoding -----
        # text_classes: list of prompt strings, e.g. ["male speech", "rain", ...]
        _, word_embeds, attn_mask = model.encode_text(text_classes)
        text_embeds = model.msc(word_embeds, model.codebook, attn_mask)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # ----- audio encoding -----
        # waveform: [num_samples]  (mono, 10 s after crop/pad)
        _, frame_embeds = model.encode_audio(waveform.unsqueeze(0))
        audio_embeds = model.msc(frame_embeds, model.codebook)

        frame_embeds = F.normalize(frame_embeds, dim=-1)   # [1, T, D]
        audio_embeds = F.normalize(audio_embeds, dim=-1)   # [1, D]

        # ----- similarity & probability -----
        frame_similarity = frame_embeds @ text_embeds.t() / model.temp   # [1, T, C]
        clip_similarity = audio_embeds @ text_embeds.t() / model.temp    # [1, C]

        frame_prob = F.softmax(frame_similarity, dim=-1) \
                        .squeeze(0).cpu().numpy().T      # [C, T]
        clip_prob = F.softmax(clip_similarity, dim=-1) \
                       .squeeze(0).cpu().numpy()         # [C]

    # ------------------------------------------------------------------
    # Print clip-level result
    # ------------------------------------------------------------------
    max_len = max(len(c) for c in text_classes)
    clip_logits = clip_similarity.squeeze(0).cpu().numpy()

    print("Class prediction (clip-level): probability")
    for cls, p, logit in zip(text_classes, clip_prob, clip_logits):
        print(f"{cls:<{max_len}} : {p:6.2f}   {logit:6.2f}")

    # ------------------------------------------------------------------
    # Plot frame-level prob heatmap
    # ------------------------------------------------------------------
    plot_prediction(
        frame_prob,
        label_list=text_classes,
        total_time=audio_len,
        figsize=(12, 2),
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_model(cfg_path: str) -> Tuple[torch.nn.Module, torch.device]:
    """Load ASE model and checkpoint from a yaml config."""
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"])
    model = ASE(config).to(device)

    ckpt_path = config["eval"]["ckpt"]
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    print(f"Model weights loaded from {ckpt_path}")
    return model, device


def load_audio(
    audio_path: str,
    device: torch.device,
    target_sr: int = 32000,
    clip_seconds: float = 10.0,
) -> Tuple[torch.Tensor, float]:
    """
    Load audio, crop/pad to `clip_seconds` and resample to `target_sr`.

    Returns:
        waveform: torch.Tensor, shape [num_samples] (mono, on device)
        audio_len: float, duration in seconds (<= clip_seconds)
    """
    waveform, sr = torchaudio.load(audio_path)  # [C, N]
    original_len_sec = waveform.shape[1] / sr
    audio_len = min(original_len_sec, clip_seconds)

    max_len_samples = int(clip_seconds * sr)

    # Crop or pad to fixed length (before resampling)
    if waveform.shape[1] > max_len_samples:
        start = random.randint(0, waveform.shape[1] - max_len_samples)
        waveform = waveform[:, start:start + max_len_samples]
    else:
        pad_len = max_len_samples - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_len))

    # Resample + mono + flatten
    resampler = T.Resample(sr, target_sr)
    waveform = resampler(waveform)          # [C, N_resampled]
    waveform = waveform.mean(dim=0)         # mono
    waveform = waveform.to(device, non_blocking=True)

    return waveform, audio_len


def plot_prediction(
    matrix: np.ndarray,
    label_list: List[str],
    figsize=(12, 4),
    total_time: float = 10.0,
    vmin: float = 0.0,
    vmax: float = 1.0,
    threshold: float = 0.05,
) -> None:
    """
    Plot frame-level class probabilities as a heatmap.

    Args:
        matrix: np.ndarray, shape [num_classes, num_frames], values in [0, 1].
        label_list: list of class names.
        total_time: total duration of the clip (seconds), used for x-axis ticks.
        vmin, vmax: color range of probabilities.
        threshold: values below this will be set to 0 for cleaner visualization.
    """
    assert matrix.shape[0] == len(label_list)

    # Optional: suppress very small probabilities to avoid noisy blobs
    display_matrix = matrix.copy()
    display_matrix[display_matrix < threshold] = 0.0

    # Add blank rows between classes to visually separate them
    rows = []
    for row in display_matrix:
        rows.append(row)
        rows.append(np.zeros_like(row))
    matrix_with_spacing = np.array(rows)

    num_frames = matrix.shape[1]
    frame_duration = total_time / num_frames

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(
        matrix_with_spacing,
        aspect="auto",
        cmap="viridis",
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Probability")

    # ----- x-axis: time in seconds -----
    tick_idx = np.arange(0, num_frames, 2)
    xticks = tick_idx
    xticklabels = [f"{(k + 0.5) * frame_duration:.2f}" for k in tick_idx]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=10)
    ax.set_xlim(-0.5, num_frames - 0.5)
    ax.set_xlabel("Time (s)")

    # ----- y-axis: class names -----
    yticks_positions = np.arange(0, len(label_list) * 2, 2)
    ax.set_yticks(yticks_positions)
    ax.set_yticklabels(label_list, fontsize=10)

    ax.grid(False)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = "settings/inference_example.yaml"
    text_classes = ["male speech", "female speech", "electric shaver"]
    audio_path = "example_audio/example.wav"

    main(cfg, text_classes, audio_path)

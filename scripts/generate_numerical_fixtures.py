#!/usr/bin/env python3
"""
Generate numerical reference fixtures for Edifice forward pass validation.

Downloads real HuggingFace models, runs forward passes on deterministic inputs,
and saves inputs + expected outputs as .safetensors files.

Requirements:
    pip install torch transformers safetensors numpy

Usage:
    python scripts/generate_numerical_fixtures.py

Output:
    test/fixtures/numerical/vit_reference.safetensors      (~600KB)
    test/fixtures/numerical/whisper_encoder_reference.safetensors (~100KB)
"""

import os
import sys
import numpy as np
import torch
from safetensors.torch import save_file


def generate_vit_fixture():
    """Generate ViT forward pass reference.

    Model: google/vit-base-patch16-224
    Input: deterministic [1, 3, 224, 224] via torch.manual_seed(42)
    Output: logits [1, 1000]
    """
    from transformers import ViTForImageClassification

    print("Loading google/vit-base-patch16-224...")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.eval()

    # Deterministic input
    torch.manual_seed(42)
    pixel_values = torch.randn(1, 3, 224, 224)

    print("Running ViT forward pass...")
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # [1, 1000]

    # Save as safetensors
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test", "fixtures", "numerical", "vit_reference.safetensors"
    )

    save_file({
        "input": pixel_values,
        "expected_logits": logits,
    }, out_path)

    print(f"Saved ViT fixture to {out_path}")
    print(f"  Input shape: {pixel_values.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits sample: {logits[0, :5].tolist()}")
    print(f"  File size: {os.path.getsize(out_path)} bytes")


def generate_whisper_encoder_fixture():
    """Generate Whisper encoder forward pass reference.

    Model: openai/whisper-base
    Input: deterministic mel spectrogram [1, 80, 100] via torch.manual_seed(42)
    Output: encoder hidden states [1, 50, 512]
        (100 mel frames -> 50 after stride-2 convolution)
    """
    from transformers import WhisperModel

    print("\nLoading openai/whisper-base...")
    model = WhisperModel.from_pretrained("openai/whisper-base")
    model.eval()

    # Deterministic mel input (shorter than full 3000 frames for fixture size)
    torch.manual_seed(42)
    mel_input = torch.randn(1, 80, 100)

    print("Running Whisper encoder forward pass...")
    with torch.no_grad():
        # Run encoder only
        encoder_output = model.encoder(mel_input)
        hidden_states = encoder_output.last_hidden_state  # [1, 50, 512]

    # Save as safetensors
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test", "fixtures", "numerical", "whisper_encoder_reference.safetensors"
    )

    save_file({
        "mel_input": mel_input,
        "expected_encoder_output": hidden_states,
    }, out_path)

    print(f"Saved Whisper encoder fixture to {out_path}")
    print(f"  Mel input shape: {mel_input.shape}")
    print(f"  Encoder output shape: {hidden_states.shape}")
    print(f"  Output sample: {hidden_states[0, 0, :5].tolist()}")
    print(f"  File size: {os.path.getsize(out_path)} bytes")


if __name__ == "__main__":
    generate_vit_fixture()
    generate_whisper_encoder_fixture()
    print("\nAll fixtures generated successfully!")

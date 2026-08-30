"""Tests for the ONNX export wrapper."""

from __future__ import annotations

import pytest
import torch

from auris_singer.export import OnnxSingerWrapper
from auris_singer.model import AurisSinger

HOP = 480


@pytest.fixture
def model(tiny_model_config):
    torch.manual_seed(0)
    return AurisSinger(**tiny_model_config).eval()


def wrapper_inputs(model: AurisSinger, batch: int = 2, s: int = 6, frames_per: int = 5):
    """A padded batch plus the noise tensors the wrapper wants."""
    torch.manual_seed(1)
    lengths = torch.tensor([s, s - 2][:batch], dtype=torch.long)
    phonemes = torch.randint(1, model.n_vocab, (batch, s))
    durations = torch.full((batch, s), frames_per, dtype=torch.long)
    t = s * frames_per  # the longer row's frame count
    f0 = torch.full((batch, t), 220.0)
    f0[:, :frames_per] = 0.0  # a leading unvoiced stretch
    voiced = (f0 > 0).float()
    energy = torch.full((batch, t), 0.1)
    speaker_ids = torch.arange(batch, dtype=torch.long) % model.n_speakers
    return {
        "phonemes": phonemes,
        "phoneme_lengths": lengths,
        "durations": durations,
        "f0": f0,
        "energy": energy,
        "voiced": voiced,
        "speaker_ids": speaker_ids,
        "noise_scale": torch.tensor(0.667),
        "z_noise": torch.zeros(batch, model.inter_channels, t),
        "source_noise": -torch.ones(batch, 1, t * HOP),
    }


def test_wrapper_matches_infer_when_the_noise_is_pinned(model, monkeypatch):
    """With every random draw forced to a constant, the two paths are the same
    computation: ``infer`` draws zeros for the prior (randn) and ``-1`` for the
    excitation (rand*2-1), and the wrapper is fed exactly those values."""
    inputs = wrapper_inputs(model)
    with torch.no_grad():
        ours = OnnxSingerWrapper(model)(**inputs)

    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    monkeypatch.setattr(torch, "rand_like", torch.zeros_like)
    theirs = model.infer(
        phonemes=inputs["phonemes"],
        phoneme_lengths=inputs["phoneme_lengths"],
        durations=inputs["durations"],
        f0=inputs["f0"],
        energy=inputs["energy"],
        voiced=inputs["voiced"],
        speaker_ids=inputs["speaker_ids"],
        noise_scale=0.667,
    )

    assert ours.shape == theirs.shape == (2, 1, inputs["f0"].size(1) * HOP)
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_wrapper_is_deterministic(model):
    inputs = wrapper_inputs(model, batch=1)
    with torch.no_grad():
        assert torch.equal(OnnxSingerWrapper(model)(**inputs), OnnxSingerWrapper(model)(**inputs))

"""Smoke tests for the Lightning training loop and the inference API."""

from __future__ import annotations

from unittest import mock

import lightning as L
import pytest
import torch

from auris_singer.data import SingingDataModule
from auris_singer.infer import Synthesizer
from auris_singer.lightning_module import AurisSingerModule
from auris_singer.text import DEFAULT_PHONEME_TABLE

AUDIO = {
    "sample_rate": 48_000,
    "n_fft": 2048,
    "hop_length": 480,
    "win_length": 2048,
    "n_mels": 80,
}
LOSS = {"mel_params": [[512, 120, 512, 40]], "envelope_kernel_sizes": [128, 256]}


@pytest.fixture
def module(tiny_model_config, tiny_discriminator_config):
    torch.manual_seed(0)
    return AurisSingerModule(
        model=tiny_model_config,
        discriminator=tiny_discriminator_config,
        audio=AUDIO,
        loss=LOSS,
        optimizer={"learning_rate": 1e-4},
        metadata={
            "symbols": DEFAULT_PHONEME_TABLE.symbols,
            "speaker_to_id": {"alice": 0, "bob": 1},
        },
    )


@pytest.fixture
def datamodule(processed_dataset):
    dm = SingingDataModule(
        processed_dataset,
        batch_size=2,
        num_workers=0,
        val_size=2,
        bucket_boundaries=[0, 200],
        pin_memory=False,
    )
    dm.setup()
    return dm


def test_configure_optimizers_returns_two_optimizers(module):
    optimizers, schedulers = module.configure_optimizers()
    assert len(optimizers) == 2 and len(schedulers) == 2
    generator_params = {id(p) for p in module.model.parameters()}
    assert all(
        id(p) in generator_params
        for group in optimizers[0].param_groups
        for p in group["params"]
    )


def test_training_updates_both_networks(module, datamodule):
    before_g = [p.detach().clone() for p in module.model.parameters()]
    before_d = [p.detach().clone() for p in module.discriminator.parameters()]

    trainer = L.Trainer(
        max_steps=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        use_distributed_sampler=False,
    )
    trainer.fit(module, datamodule=datamodule)

    assert trainer.global_step == 2
    assert any(
        not torch.equal(a, b) for a, b in zip(before_g, module.model.parameters())
    ), "generator did not update"
    assert any(
        not torch.equal(a, b) for a, b in zip(before_d, module.discriminator.parameters())
    ), "discriminator did not update"
    assert all(torch.isfinite(p).all() for p in module.model.parameters())


def test_validation_runs_the_full_inference_path(module, datamodule):
    trainer = L.Trainer(
        max_steps=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_val_batches=2,
        val_check_interval=1,  # validate after the single training step
        use_distributed_sampler=False,
    )
    trainer.fit(module, datamodule=datamodule)
    assert torch.isfinite(trainer.callback_metrics["val/mel"])


def test_checkpoint_roundtrip_synthesizes_audio(module, datamodule, tmp_path):
    trainer = L.Trainer(
        max_steps=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        use_distributed_sampler=False,
    )
    trainer.fit(module, datamodule=datamodule)

    checkpoint = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint)

    synthesizer = Synthesizer.from_checkpoint(checkpoint)
    assert synthesizer.resolve_speaker("bob") == 1
    phonemes = ["<sil>", "k", "o", "ɴ", "i"]
    durations = [4, 5, 6, 3, 4]
    n_frames = sum(durations)
    wav = synthesizer.synthesize(
        phonemes=phonemes,
        durations=durations,
        f0=[220.0] * n_frames,
        energy=[0.1] * n_frames,
        speaker="alice",
    )
    assert wav.shape == (n_frames * AUDIO["hop_length"],)
    assert wav.dtype.name == "float32"


class _FakeExperiment:
    def __init__(self):
        self.calls: list[str] = []

    def add_audio(self, tag, audio, step, sample_rate):
        self.calls.append(tag)


class _FakeLogger:
    def __init__(self):
        self.experiment = _FakeExperiment()


def test_reference_audio_is_logged_once_regardless_of_epoch(module):
    """Step-based validation means epoch 0 is long gone on the first pass."""
    logger = _FakeLogger()
    wav = torch.zeros(1, 480)
    with mock.patch.object(
        type(module), "logger", new_callable=mock.PropertyMock, return_value=logger
    ):
        module._log_audio(0, wav, wav)
        module._log_audio(0, wav, wav)
        module._log_audio(1, wav, wav)

    tags = logger.experiment.calls
    assert tags.count("val/0/reference") == 1, "reference must not be re-logged"
    assert tags.count("val/0/generated") == 2, "generated audio changes every pass"
    assert "val/1/reference" in tags


def test_synthesize_validates_its_inputs(module):
    synthesizer = Synthesizer(module)
    with pytest.raises(ValueError, match="durations has"):
        synthesizer.synthesize(["a", "i"], [3], [220.0] * 3, [0.1] * 3)
    with pytest.raises(ValueError, match="sum\\(durations\\)"):
        synthesizer.synthesize(["a", "i"], [3, 3], [220.0] * 5, [0.1] * 5)
    with pytest.raises(ValueError, match="not in the table"):
        synthesizer.synthesize(["a", "zzz"], [2, 2], [220.0] * 4, [0.1] * 4)
    with pytest.raises(KeyError, match="unknown speaker"):
        synthesizer.resolve_speaker("nobody")

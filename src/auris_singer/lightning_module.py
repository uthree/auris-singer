"""Lightning module wiring the generator, the discriminators and the losses.

GAN training needs two optimizers stepped in a fixed order, so the module runs
with ``automatic_optimization = False``.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F

from auris_singer.losses import (
    EnvelopeLoss,
    MultiParamMelLoss,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    kl_loss,
)
from auris_singer.model import AurisSinger
from auris_singer.modules.discriminator import Discriminator
from auris_singer.utils.audio import mel_spectrogram
from auris_singer.utils.masks import slice_segments

__all__ = ["AurisSingerModule"]


class AurisSingerModule(L.LightningModule):
    """Training wrapper.

    Args:
        model: keyword arguments for :class:`~auris_singer.model.AurisSinger`.
        discriminator: keyword arguments for
            :class:`~auris_singer.modules.discriminator.Discriminator`.
        audio: spectrogram settings, used for logging and the mel losses.
        loss: loss weights and per-loss settings.
        optimizer: learning rate, betas, weight decay and LR decay.
        metadata: dataset metadata (phoneme symbols, speaker map) stored in the
            checkpoint so inference needs nothing but the checkpoint file.
    """

    def __init__(
        self,
        model: dict[str, Any],
        discriminator: dict[str, Any] | None = None,
        audio: dict[str, Any] | None = None,
        loss: dict[str, Any] | None = None,
        optimizer: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        model = dict(model)
        audio = dict(audio or {})
        loss = dict(loss or {})
        optimizer = dict(optimizer or {})
        discriminator = dict(discriminator or {})

        self.sample_rate = int(audio.get("sample_rate", model.get("sample_rate", 48_000)))
        self.hop_length = int(audio.get("hop_length", model.get("hop_length", 480)))
        self.n_fft = int(audio.get("n_fft", 2048))
        self.win_length = int(audio.get("win_length", self.n_fft))
        self.n_mels = int(audio.get("n_mels", 128))
        self.f_min = float(audio.get("f_min", 0.0))
        self.f_max = audio.get("f_max", None)

        self.model = AurisSinger(**model)
        discriminator.setdefault("n_speakers", model.get("n_speakers", 1))
        self.discriminator = Discriminator(**discriminator)

        self.envelope_loss = EnvelopeLoss(
            kernel_sizes=tuple(loss.get("envelope_kernel_sizes", (128, 256, 512, 1024)))
        )
        mel_params = loss.get(
            "mel_params",
            (
                (512, 120, 512, 40),
                (1024, 240, 1024, 80),
                (2048, 480, 2048, 128),
                (4096, 960, 4096, 160),
            ),
        )
        self.mel_loss = MultiParamMelLoss(
            sample_rate=self.sample_rate,
            params=tuple(tuple(p) for p in mel_params),
            f_min=self.f_min,
            f_max=self.f_max,
        )

        self.weights = {
            "mel": float(loss.get("mel", 45.0)),
            "kl": float(loss.get("kl", 1.0)),
            "kl_aux": float(loss.get("kl_aux", 1.0)),
            "feature_matching": float(loss.get("feature_matching", 1.0)),
            "envelope": float(loss.get("envelope", 10.0)),
            "adversarial": float(loss.get("adversarial", 1.0)),
        }

        self.learning_rate = float(optimizer.get("learning_rate", 2e-4))
        self.betas = tuple(optimizer.get("betas", (0.8, 0.99)))
        self.eps = float(optimizer.get("eps", 1e-9))
        self.weight_decay = float(optimizer.get("weight_decay", 0.0))
        self.lr_decay = float(optimizer.get("lr_decay", 0.999875))
        self.grad_clip = float(optimizer.get("grad_clip", 0.0))

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.lr_decay)
        sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=self.lr_decay)
        return [opt_g, opt_d], [sch_g, sch_d]

    # ------------------------------------------------------------------
    def _log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        return mel_spectrogram(
            wav.float(),
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
        )

    def _clip(self, module: torch.nn.Module) -> None:
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.grad_clip)

    # ------------------------------------------------------------------
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        opt_g, opt_d = self.optimizers()
        speaker_ids = batch["speaker_ids"]

        out = self.model(
            phonemes=batch["phonemes"],
            phoneme_lengths=batch["phoneme_lengths"],
            spec=batch["spec"],
            spec_lengths=batch["spec_lengths"],
            f0=batch["f0"],
            energy=batch["energy"],
            voiced=batch["voiced"],
            speaker_ids=speaker_ids,
        )
        wav_hat = out["wav_hat"]
        segment_samples = wav_hat.size(-1)
        wav_real = slice_segments(
            batch["wav"], out["slice_ids"] * self.hop_length, segment_samples
        )

        # --- discriminator ------------------------------------------------
        real_out, _ = self.discriminator(wav_real, speaker_ids)
        fake_out, _ = self.discriminator(wav_hat.detach(), speaker_ids)
        loss_disc, _, _ = discriminator_loss(real_out, fake_out)

        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(loss_disc)
        self._clip(self.discriminator)
        opt_d.step()

        # --- generator ----------------------------------------------------
        real_out, real_fmap = self.discriminator(wav_real, speaker_ids)
        fake_out, fake_fmap = self.discriminator(wav_hat, speaker_ids)

        loss_adv, _ = generator_adversarial_loss(fake_out)
        loss_fm = feature_matching_loss(real_fmap, fake_fmap)
        loss_mel = self.mel_loss(wav_real, wav_hat)
        loss_env = self.envelope_loss(wav_real, wav_hat)
        loss_kl = kl_loss(
            out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["y_mask"]
        )
        # The auxiliary term is the objective monotonic alignment search
        # maximizes; keeping it in the loss stops the alignment prior from
        # drifting away from the refined prior.
        loss_kl_aux = kl_loss(
            out["z_p"],
            out["logs_q"],
            out["m_p0_frame"],
            out["logs_p0_frame"],
            out["y_mask"],
        )

        loss_gen = (
            self.weights["adversarial"] * loss_adv
            + self.weights["feature_matching"] * loss_fm
            + self.weights["mel"] * loss_mel
            + self.weights["envelope"] * loss_env
            + self.weights["kl"] * loss_kl
            + self.weights["kl_aux"] * loss_kl_aux
        )

        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(loss_gen)
        self._clip(self.model)
        opt_g.step()

        self.log_dict(
            {
                "train/loss_disc": loss_disc,
                "train/loss_gen": loss_gen,
                "train/adv": loss_adv,
                "train/feature_matching": loss_fm,
                "train/mel": loss_mel,
                "train/envelope": loss_env,
                "train/kl": loss_kl,
                "train/kl_aux": loss_kl_aux,
            },
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )
        self.log("train/loss", loss_gen, prog_bar=True, on_step=True, on_epoch=False)

    def on_train_epoch_end(self) -> None:
        for scheduler in self.lr_schedulers():
            scheduler.step()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        speaker_ids = batch["speaker_ids"]
        out = self.model(
            phonemes=batch["phonemes"],
            phoneme_lengths=batch["phoneme_lengths"],
            spec=batch["spec"],
            spec_lengths=batch["spec_lengths"],
            f0=batch["f0"],
            energy=batch["energy"],
            voiced=batch["voiced"],
            speaker_ids=speaker_ids,
        )
        durations = out["durations"].round().long()

        wav_hat = self.model.infer(
            phonemes=batch["phonemes"],
            phoneme_lengths=batch["phoneme_lengths"],
            durations=durations,
            f0=batch["f0"],
            energy=batch["energy"],
            voiced=batch["voiced"],
            speaker_ids=speaker_ids,
        )
        wav_real = batch["wav"]
        length = min(wav_hat.size(-1), wav_real.size(-1))
        mel_hat = self._log_mel(wav_hat[..., :length])
        mel_real = self._log_mel(wav_real[..., :length])
        loss_mel = F.l1_loss(mel_hat, mel_real)

        self.log("val/mel", loss_mel, prog_bar=True, on_epoch=True, sync_dist=True)
        if batch_idx < 4:
            self._log_audio(batch_idx, wav_hat[0, :, :length], wav_real[0, :, :length])

    def _log_audio(
        self, index: int, wav_hat: torch.Tensor, wav_real: torch.Tensor
    ) -> None:
        logger = self.logger
        experiment = getattr(logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_audio"):
            return
        experiment.add_audio(
            f"val/{index}/generated",
            wav_hat.float().cpu(),
            self.global_step,
            self.sample_rate,
        )
        if self.current_epoch == 0:
            experiment.add_audio(
                f"val/{index}/reference",
                wav_real.float().cpu(),
                self.global_step,
                self.sample_rate,
            )

"""ONNX export of the inference path.

The graph exported here is :meth:`AurisSinger.infer` rewritten as a pure
function: every stochastic draw becomes an input.  A caller that feeds the
same noise twice gets the same waveform twice — which is what a DAW needs for
reproducible renders, and what makes the export verifiable against PyTorch at
all (with graph-internal random ops the two runtimes could only ever be
compared statistically).

:class:`OnnxSingerWrapper` is that pure function as an ``nn.Module``;
:func:`export_onnx` traces it into an ``.onnx`` file with dynamic sequence
lengths and embeds the checkpoint's metadata (phoneme table, speaker map,
audio parameters) so the consumer needs nothing but the one file.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from auris_singer.model import AurisSinger
from auris_singer.utils.masks import sequence_mask

__all__ = ["OnnxSingerWrapper"]


class OnnxSingerWrapper(nn.Module):
    """The inference path as a pure function of tensors.

    Differences from :meth:`AurisSinger.infer`, all in the name of a clean
    ONNX graph:

    * ``voiced`` is required — deriving it from ``f0`` would silently voice
      the consonant frames of a front-end that writes pitch as a contour
      (auris-studio does exactly that);
    * the prior sample and the excitation noise are inputs (``z_noise``,
      ``source_noise``) instead of internal draws;
    * ``sum(durations)`` must equal ``f0.size(-1)`` — the wrapper does not
      trim the curves the way ``infer`` does, because data-dependent slicing
      does not belong in a traced graph.
    """

    def __init__(self, model: AurisSinger):
        super().__init__()
        self.model = model

    def forward(
        self,
        phonemes: torch.Tensor,
        phoneme_lengths: torch.Tensor,
        durations: torch.Tensor,
        f0: torch.Tensor,
        energy: torch.Tensor,
        voiced: torch.Tensor,
        speaker_ids: torch.Tensor,
        noise_scale: torch.Tensor,
        z_noise: torch.Tensor,
        source_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            phonemes: ``(B, S)`` phoneme ids, int64.
            phoneme_lengths: ``(B,)`` int64.
            durations: ``(B, S)`` frames per phoneme, int64; each row must sum
                to ``T``.
            f0: ``(B, T)`` f0 in Hz, float32; 0 on unvoiced and silent frames.
            energy: ``(B, T)`` linear RMS energy, float32.
            voiced: ``(B, T)`` float32, 1.0 on voiced frames.
            speaker_ids: ``(B,)`` int64.
            noise_scale: scalar float32 — the prior sampling temperature.
            z_noise: ``(B, inter_channels, T)`` standard normal draws, float32.
            source_noise: ``(B, 1, T * hop_length)`` uniform noise on
                ``[-1, 1]``, float32.

        Returns:
            ``(B, 1, T * hop_length)`` waveform, float32.
        """
        model = self.model
        g = model.speaker_embedding(speaker_ids).unsqueeze(-1)

        x, _, _, x_mask = model.text_encoder(phonemes, phoneme_lengths, g=g)

        durations = durations.to(torch.long) * x_mask.squeeze(1).long()
        y_lengths = durations.sum(dim=1).clamp(min=1)
        y_mask = sequence_mask(y_lengths, f0.size(-1)).unsqueeze(1).to(x.dtype)

        attn = model._path_from_durations(durations, x_mask, y_mask)
        x_frame = torch.matmul(x, attn)

        f0 = f0.unsqueeze(1)
        energy = energy.unsqueeze(1)
        voiced = voiced.unsqueeze(1)

        m_p, logs_p = model.prior_encoder(
            x_frame, y_mask, f0=f0, energy=energy, voiced=voiced, g=g
        )
        z_p = m_p + z_noise * torch.exp(logs_p) * noise_scale
        z = model.flow(z_p, y_mask, g=g, reverse=True)

        source = model.generator.source_generator(f0, energy, voiced, noise=source_noise)
        wav, _ = model.generator(z * y_mask, f0, energy, voiced, g=g, source=source)
        return wav

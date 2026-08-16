# Inference

The model is not given a score. It is given a phoneme sequence, an integer
duration per phoneme, and frame-level f0 and energy curves. Converting a score
into those curves is the DAW front-end's job and is out of scope for this
repository.

## Command line

```bash
uv run python scripts/infer.py \
    --checkpoint runs/base/checkpoints/last.ckpt \
    --input score.json \
    --output out.wav \
    --device cuda
```

`score.json`:

```json
{
  "speaker": "my_singer",
  "phonemes": ["<sil>", "k", "o", "ɴ", "ɲ", "i", "tɕ", "i", "w", "a", "<sil>"],
  "durations": [20, 6, 10, 8, 6, 12, 6, 10, 6, 25, 20],
  "f0":     [0.0, 0.0, 220.0, 220.4, "..."],
  "energy": [0.0, 0.01, 0.08, 0.09, "..."]
}
```

Rules:

* `durations` counts **frames** per phoneme — 100 frames per second at 48 kHz
  with the default hop of 480 samples. It must be the same length as
  `phonemes`.
* `f0` and `energy` must each have exactly `sum(durations)` entries.
* `f0` is in Hz; `0` marks an unvoiced frame. `voiced` may be supplied
  explicitly as an optional array of the same length; otherwise it is derived
  from `f0`.
* `energy` is linear RMS, on the same scale the preprocessing pipeline
  produced (roughly `0.0`–`0.5` for peak-normalized audio).
* `speaker` may be a name from the training set or an integer id.

## Python API

```python
from auris_singer.infer import Synthesizer

synth = Synthesizer.from_checkpoint("runs/base/checkpoints/last.ckpt", device="cuda")

wav = synth.synthesize(
    phonemes=["<sil>", "a", "i", "<sil>"],
    durations=[20, 50, 50, 20],
    f0=[0.0] * 20 + [440.0] * 50 + [493.9] * 50 + [0.0] * 20,
    energy=[0.0] * 20 + [0.15] * 100 + [0.0] * 20,
    speaker="my_singer",
    noise_scale=0.667,
)   # -> float32 numpy array at synth.sample_rate
```

The checkpoint carries the phoneme table and the speaker map, so nothing else
needs to be loaded. `synth.speaker_to_id` lists the available speakers.

`noise_scale` is the sampling temperature of the prior. Lower values give a
flatter, more deterministic delivery; `0.0` makes synthesis deterministic.

## Getting phonemes from Japanese text

```python
from auris_singer.text import JapaneseFrontend

phonemes = JapaneseFrontend().g2p("こんにちは")
# ['<sil>', 'k', 'o', 'ɴ', 'ɲ', 'i', 'tɕ', 'i', 'w', 'a', '<sil>']
```

Durations still have to come from somewhere — the front-end only produces the
symbol sequence.

## Control notes

Pitch and loudness reach the decoder **only** through the source signal (see
[architecture.md](architecture.md#source-signal-refinegan-style)), so they are
directly controllable:

* transposing `f0` transposes the output without touching timbre;
* scaling `energy` scales loudness and, because the excitation amplitude
  changes with it, the accompanying change in vocal effort;
* setting `f0` to 0 over a span makes that span unvoiced (breath, whisper-like
  consonants).

Very large deviations from the training distribution — an octave above anything
in the data, say — will degrade quality; the prior is still conditioned on f0
and energy and has only seen the training range.

## Exporting for deployment

```python
module = AurisSingerModule.load_from_checkpoint("last.ckpt")
module.model.remove_weight_norm()   # fold weight norm into the weights
```

This is a one-way operation; do it on a copy loaded for inference, not on a
checkpoint you intend to keep training.

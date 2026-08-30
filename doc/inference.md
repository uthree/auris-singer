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

## Exporting to ONNX

```bash
uv pip install -e '.[export]'
uv run python scripts/export_onnx.py \
    --checkpoint runs/base/checkpoints/last.ckpt --output runs/base/model.onnx
```

This writes `model.onnx` plus a `model.json` sidecar and, unless `--no-verify`
is given, checks the graph against PyTorch with onnxruntime at input sizes the
trace never saw. Weight norm is folded into the weights as part of the export
(`remove_weight_norm()` — a one-way operation, which is why the script loads
its own copy of the checkpoint).

The graph is the same computation `Synthesizer.synthesize` runs, made a pure
function: the random draws are inputs, so a caller that seeds its own
generator gets bit-identical renders. All inputs are required.

| input | shape | dtype | |
| --- | --- | --- | --- |
| `phonemes` | `(B, S)` | int64 | ids into the phoneme table in the metadata |
| `phoneme_lengths` | `(B,)` | int64 | valid entries per row |
| `durations` | `(B, S)` | int64 | frames per phoneme; each row sums to `T` |
| `f0` | `(B, T)` | float32 | Hz; 0 on unvoiced and silent frames |
| `energy` | `(B, T)` | float32 | linear RMS, as in the score input above |
| `voiced` | `(B, T)` | float32 | 1.0 on voiced frames |
| `speaker_ids` | `(B,)` | int64 | |
| `noise_scale` | scalar | float32 | prior sampling temperature |
| `z_noise` | `(B, inter_channels, T)` | float32 | standard normal draws |
| `source_noise` | `(B, 1, T * hop_length)` | float32 | uniform on [-1, 1] |

Outputs: `wav` `(B, 1, T * hop_length)` float32, and `source`, the excitation
signal at the same shape — a diagnostics output that a runtime asked only for
`wav` never computes.

Two contract points that differ from the Python API:

* **`voiced` is required, not derived from `f0`.** A DAW front-end that
  writes pitch as a contour puts real f0 values on unvoiced consonant
  frames; deriving voicing from `f0 > 0` would silently voice them. Decide
  voicing from the phoneme class and say so explicitly.
* **`sum(durations)` must equal `T` exactly** — the graph does not trim the
  curves the way `Synthesizer.synthesize` does.

The phoneme table, the speaker map and the audio parameters ride along as
JSON, both under the `auris_singer` key of the ONNX `metadata_props` and in
the `.json` sidecar:

```json
{
  "format_version": 1,
  "sample_rate": 48000,
  "hop_length": 480,
  "inter_channels": 192,
  "n_speakers": 2,
  "f0_min": 40.0,
  "symbols": ["<pad>", "<unk>", "<sil>", "..."],
  "speaker_to_id": {"my_singer": 0}
}
```

`inter_channels` is there so the caller can shape `z_noise` without knowing
the model config; `symbols` maps IPA strings to the ids `phonemes` wants
(index in the list = id).

### Voice card

Presentational metadata — what a host application shows to a person browsing
voices, as opposed to what it feeds the model — travels in the same JSON under
the `voice` key:

```bash
uv run python scripts/export_onnx.py --checkpoint last.ckpt --output ritsu.onnx \
    --voice-card card.json --portrait ritsu.png
```

`card.json` is a free-form JSON object; these field names are the convention a
UI can rely on:

```json
{
  "name": "波音リツ",
  "description": "Strong low-range female voice. 107 songs, 4.4 h.",
  "author": "...",
  "version": "1.0",
  "license": "Namine Ritsu singing DB terms; fine-tuning to other voices prohibited",
  "credits": ["波音リツ", "カノン"],
  "url": "https://..."
}
```

`--portrait` embeds a character image (png/jpeg/webp, at most 8 MB) as
`voice.portrait = {"mime": ..., "base64": ...}` — decode the base64 to get the
image bytes back. Everything, artwork included, lives inside the one `.onnx`
file (and its `.json` sidecar), so a published model file carries its own
name, description, credit line and 立ち絵 with no companion archive to lose.

From Rust, the [ort](https://ort.pyke.io) crate runs the file as-is on CPU;
request only the `wav` output. Renders are reproducible: same inputs, same
noise, same waveform. (Across *runtimes* the match is exact except for the
excitation's impulse timing, where float32 rounding can shift an impulse by
one sample — inaudible, and training's random phase offset makes the model
indifferent to it by construction.)

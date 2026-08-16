# Preprocessing

Preprocessing turns a corpus of audio and transcripts into one `.npz` file per
utterance holding every feature training needs.

```bash
uv run python scripts/preprocess.py --config configs/preprocess/generic_wav_text.yml
```

Any config value can be overridden on the command line:

```bash
uv run python scripts/preprocess.py --config configs/preprocess/generic_wav_text.yml \
    dataset.output_dir=data/other f0.device=cpu
```

## Input layout

The shipped config expects one directory of audio and one of transcripts per
speaker:

```
data/raw/my_singer/wav/song01.wav
data/raw/my_singer/text/song01.txt
```

Transcripts are matched by stem, and lookup recurses into subdirectories.
Add one entry per speaker under `dataset.sources`:

```yaml
dataset:
  sources:
    - name: singer_a
      wav_dir: data/raw/singer_a/wav
      text_dir: data/raw/singer_a/text
    - name: singer_b
      wav_dir: data/raw/singer_b/wav
      text_dir: data/raw/singer_b/text
```

Durations are **not** required. Training recovers the phoneme/frame alignment
with monotonic alignment search (see [architecture.md](architecture.md)).

## Output layout

```
data/processed/
  my_singer/song01.npz
  metadata.jsonl      # one JSON record per utterance
  speakers.json       # speaker name -> id
  phonemes.json       # the IPA symbol table used
  audio_config.json   # sample rate / STFT settings the features were built with
```

Each `.npz` contains:

| key | dtype | shape | contents |
| --- | --- | --- | --- |
| `wav` | int16 | `(T · hop,)` | resampled, peak-normalized waveform |
| `phonemes` | int32 | `(S,)` | IPA phoneme ids |
| `f0` | float32 | `(T,)` | Hz, exactly 0 on unvoiced frames |
| `energy` | float32 | `(T,)` | per-frame RMS |
| `voiced` | uint8 | `(T,)` | voiced flag |

The linear spectrogram is deliberately **not** cached — recomputing it in the
dataloader is cheap and keeps the dataset roughly 4× smaller.

Every frame-level feature shares one grid: the waveform is truncated to a whole
number of frames, and `T = len(wav) // hop_length`.

## Configuration reference

```yaml
audio:
  sample_rate: 48000     # everything is resampled to this
  n_fft: 2048
  hop_length: 480        # 100 frames per second
  win_length: 2048
  peak_normalize: true   # scale each utterance to `peak`
  peak: 0.95
  min_seconds: 0.5       # shorter utterances are skipped
  max_seconds: 20.0      # longer ones are truncated

f0:
  device: cuda           # FCPE runs here; use cpu if no GPU
  f0_min: 40.0
  f0_max: 1600.0         # high enough for soprano singing
  threshold: 0.006       # FCPE voicing threshold
  decoder_mode: local_argmax

text:
  language: ja           # "ja" runs jpreprocess; "ipa" takes IPA directly
  phoneme_table: null    # path to a saved table, or null for the built-in one
  options:
    add_boundary_silence: true

dataset:
  output_dir: data/processed
  sources: [...]

num_workers: 8           # threads for audio loading and g2p
```

The `audio` block **must** match the `audio` block of the training config; the
test suite checks this for the shipped configs.

## Text front-end

`text.language: ja` runs [`jpreprocess`](https://github.com/jpreprocess/jpreprocess)
(an OpenJTalk rewrite) and maps its romanized phoneme set to IPA. The
dictionary (~30 MB) is downloaded on first use.

`text.language: ipa` treats each transcript as a whitespace-separated IPA
sequence, e.g. `k o ɴ ɲ i tɕ i w a`. IPA symbols are multi-character, so the
spaces are required.

The symbol table lives in
[`text/ipa.py`](../src/auris_singer/text/ipa.py) and is fixed in code rather
than derived from the data, so phoneme ids stay stable across datasets and
checkpoints. Symbols missing from the table are logged and encoded as `<unk>`.

To add a language, write a front-end returning IPA symbols and register it in
`text/__init__.py::get_frontend`.

## Recipe: JSUT-song

[JSUT-song](https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song)
is a convenient small corpus to start from: 27 children's songs by one singer,
already at 48 kHz, about 25 minutes, CC BY-SA 4.0.

```bash
curl -O https://ss-takashi.sakura.ne.jp/corpus/jsut-song_ver1.zip
curl -O https://ss-takashi.sakura.ne.jp/corpus/jsut-song_label.zip
unzip -q jsut-song_ver1.zip -d data/raw && unzip -q jsut-song_label.zip -d data/raw
```

The recordings are 30–80 s long and there are no plain-text transcripts, so a
preparation step is needed:

```bash
uv run python scripts/prepare_jsut_song.py \
    --wav-dir data/raw/jsut-song_ver1/child_song/wav \
    --label-dir data/raw/todai_child \
    --output data/raw/jsut_song
```

This splits each recording into phrases at the pauses its label marks, and
writes the phoneme sequence out as an IPA transcript. Songs that sustain a
legato line with no usable pause are cut at a consonant onset instead, which in
a CV language is a syllable boundary. Loudness is normalized per song, not per
phrase, so phrase-to-phrase dynamics survive — which is why
`configs/preprocess/jsut_song.yml` sets `audio.peak_normalize: false`.

On this corpus it yields ~260 phrases of 0.8–8 s, 21.6 minutes total.

```bash
uv run python scripts/preprocess.py --config configs/preprocess/jsut_song.yml
uv run python scripts/train.py --config configs/train/small.yml \
    data.root=data/processed/jsut_song data.batch_size=12
```

Note that 20 minutes of a single singer is enough to check that the pipeline
learns, not to produce a good voice.

## Skipped utterances

An utterance is skipped when it has no transcript, produces no phonemes, is
shorter than `min_seconds`, or has fewer frames than phonemes (monotonic
alignment search needs at least one frame per phoneme). The run prints a
summary of skip reasons at the end.

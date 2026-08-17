# Training

```bash
uv run python scripts/train.py --config configs/train/base.yml
```

Long runs should go in tmux so they survive a disconnect:

```bash
tmux new-session -d -s train "cd /path/to/auris-singer && uv run python scripts/train.py --config configs/train/base.yml"
```

```bash
tmux attach -t train
```

Overrides use the same dotlist syntax as preprocessing:

```bash
uv run python scripts/train.py --config configs/train/base.yml \
    data.root=data/processed data.batch_size=8 trainer.precision=32-true
```

Resume from a checkpoint:

```bash
uv run python scripts/train.py --config configs/train/base.yml --resume runs/base/checkpoints/last.ckpt
```

## Presets

`configs/train/presets.yml` holds the model-size presets; a training config
selects one through its `defaults` block:

```yaml
defaults:
  - presets.yml: base
```

Keys set in the including file win over the preset, so a config can select
`base` and still change, say, `model.generator.upsample_initial_channel`.

| Preset | hidden | encoder layers | decoder channels |
| --- | --- | --- | --- |
| `small` | 192 | 4 | 256 |
| `base` | 256 | 6 | 512 |

`n_vocab` and `n_speakers` are filled in from the preprocessed dataset by
`scripts/train.py`, so they never appear in a config file.

## Optimization

Two AdamW optimizers (generator and discriminators) are stepped manually in
that order, so the module runs with `automatic_optimization = False`. Each has
an `ExponentialLR` schedule stepped once per **epoch**.

Defaults follow VITS: `lr = 2e-4`, `betas = (0.8, 0.99)`, `lr_decay = 0.999875`.
`optimizer.grad_clip` is off by default; set it to e.g. `10.0` if the
discriminator loss spikes.

`bf16-mixed` precision is the default and is recommended over `16-mixed`: the
spectrogram and mel losses are computed in float32 either way, but bf16 avoids
gradient-scaler interactions with manual optimization.

## Batching

The train loader uses `DistributedBucketSampler`, which groups utterances into
frame-count buckets so a batch contains similar-length clips and padding waste
stays small. It is already distribution-aware, so the trainer is created with
`use_distributed_sampler=False` — keep that if you write your own entry point.

`data.max_frames` bounds the memory of a batch (1200 frames = 12 s), and
`data.bucket_boundaries` must cover up to `max_frames`.

## Losses

See [architecture.md](architecture.md#losses) for what each term does. The
weights live under `loss:` in the training config:

```yaml
loss:
  mel: 45.0
  kl: 1.0
  kl_aux: 0.2             # alignment statistic only; 1.0 doubles KL pressure
  feature_matching: 1.0
  envelope: 10.0
  adversarial: 1.0
  kl_free_bits: 0.02      # nats per latent channel per frame; 0 disables
  kl_warmup_steps: 10000  # optimizer steps to ramp the KL weight from 0 to 1
```

`kl_free_bits` and `kl_warmup_steps` guard against posterior collapse, which
this architecture invites and which is worth understanding before touching any
KL weight — see
[architecture.md](architecture.md#posterior-collapse-the-failure-mode-this-design-invites).

`envelope` is the one weight with no precedent to copy: the envelope L1 is
computed on raw waveform amplitudes, so its raw magnitude is small compared to
the mel term. 10.0 puts it in a comparable range; lower it if the model
underfits spectral detail, raise it if dynamics sound flat.

## Step counting

Two units are in play, and they are **not** the same:

* `trainer.max_steps` and `checkpoint.every_n_train_steps` count **optimizer
  steps**. Manual optimization with two optimizers means Lightning advances
  `global_step` twice per batch, so these are 2× the number of batches.
* `trainer.val_check_interval` and `trainer.log_every_n_steps` count
  **batches**.

Set `val_check_interval` well below `max_steps / 2`, or validation never runs.

## Monitoring

TensorBoard logs go to `log_dir`:

```bash
uv run tensorboard --logdir runs/base/logs
```

Training scalars: `train/loss_disc`, `train/loss_gen`, and the individual
`train/mel`, `train/envelope`, `train/kl`, `train/kl_aux`, `train/adv`,
`train/feature_matching`.

Validation synthesizes audio through the **full inference path** (durations
from MAS, then prior sampling and flow inversion) rather than reconstructing
from the posterior, so what you hear is what synthesis will sound like. Audio
for the first `validation.log_audio_batches` utterances is logged, along with
the metrics below.

What to expect: `train/mel` should fall steadily; `train/kl` typically rises
early as the flow starts using its capacity, then settles. The adversarial
losses oscillate — that is normal. Alignment quality is the thing to watch
early: if `train/kl_aux` plateaus high, MAS is probably not finding a sensible
alignment, usually because of noisy transcripts or leading/trailing silence
that the `<sil>` boundary tokens do not cover.

## Validation metrics

`val/mel` measures overall spectral accuracy, but it says little about the one
thing this architecture is built around: f0 and energy reach the decoder *only*
through the excitation signal, so the question is whether the output actually
follows the curves it was given. The generated waveform is therefore
re-analysed — with FCPE for pitch and frame RMS for loudness — and compared
against the input curves.

| Metric | Meaning | Good direction |
| --- | --- | --- |
| `val/f0_rmse_cent` | pitch error on frames both sides call voiced, in cents (1200 = octave) | ↓ under ~50 is in tune |
| `val/f0_accuracy` | fraction of those frames within `tolerance_cents` (default 50 = a quarter tone) | ↑ toward 1.0 |
| `val/f0_corr` | correlation of the two pitch contours in cents | ↑ toward 1.0 |
| `val/vuv_error` | fraction of frames whose voiced/unvoiced decision disagrees | ↓ toward 0 |
| `val/voiced_ratio_error` | signed difference in overall voiced fraction | negative = too breathy, positive = too buzzy |
| `val/energy_rmse_db` | level error of the loudness envelope | ↓ |
| `val/energy_bias_db` | signed mean level error | negative = systematically quiet |
| `val/energy_corr` | correlation of the two envelopes in dB | ↑ toward 1.0 |
| `val/latent_usage` | how much worse the decoder gets when `z` is permuted in time | ↑; near 0 means posterior collapse |

`val/latent_usage` is the one that catches the failure mode specific to this
architecture. Phonetic content reaches the decoder only through `z`, while pitch
and loudness arrive through the excitation, so a collapsed model sings the right
notes with no intelligible words — and every other metric here still looks
excellent. If permuting `z` along time costs the decoder nothing, the latent is
dead; see [architecture.md](architecture.md#posterior-collapse-the-failure-mode-this-design-invites).
Both decoder passes reuse the *same* excitation, so a collapsed model scores a
true 0 rather than the noise floor a freshly drawn excitation would leave.
`train/posterior_sigma` drifting above 1 while `train/kl` heads to 0 is the same
story seen from the training side.

`f0_rmse_cent` and `energy_bias_db` are the two to watch. A model that sounds
plausible but ignores the source shows up here as a high pitch RMSE with a
perfectly reasonable `val/mel` — the failure mode a mel loss cannot see. The
signed terms separate mistakes an RMSE hides: a consistently quiet output and a
noisy-but-centred one have the same `energy_rmse_db`.

Metrics with no frames to average over (nothing voiced, everything silent) are
NaN and are dropped rather than logged as 0, so an undefined metric never looks
like a perfect score.

Configure under `validation:`:

```yaml
validation:
  pitch_metrics: true      # false skips FCPE and logs energy metrics only
  f0_min: 40.0
  f0_max: 1600.0
  tolerance_cents: 50.0
  log_audio_batches: 4
```

FCPE is loaded lazily on the first validation pass. If it cannot be loaded, the
pitch metrics are disabled with a warning and training continues.

### Checking that control actually works

The validation metrics compare the output against the **ground-truth** curves of
the reference audio, which is a necessary check but not a sufficient one: a
model that ignored the excitation and reconstructed the utterance from the
latent alone would still score well, because the reference happens to have
exactly that pitch.

`scripts/check_source_control.py` closes that gap by re-synthesizing one
utterance with **modified** curves and measuring whether the output followed:

```bash
uv run python scripts/check_source_control.py \
    --checkpoint runs/base/checkpoints/last.ckpt \
    --dataset data/processed/jsut_song \
    --output-dir runs/base/control_check
```

Measured on the 40k-step JSUT-song run:

```
condition             f0 err (cent)   f0 acc  f0 corr   energy bias (dB)  energy corr
reference                      30.7    0.948   0.9941               0.21        0.960
pitch_down_5st                 41.7    0.923   0.9895              -0.73        0.960
pitch_down_2st                 31.9    0.934   0.9939              -0.02        0.959
pitch_up_3st                   25.7    0.947   0.9959               0.16        0.961
pitch_up_7st                   34.1    0.934   0.9929              -1.54        0.923
energy_x0.5                    35.3    0.953   0.9920               1.06        0.942
energy_x2                      31.9    0.948   0.9938              -0.95        0.955
```

The error should stay roughly flat as the curve moves: that means the output
tracked the *new* target, not the one it was trained on. A sharp rise under
transposition means the model is reconstructing from memory instead of being
controlled. Expect degradation far outside the training pitch range — `+7st` on
a corpus that never goes that high is not a fair test; above, the ±7 semitone
extremes cost a few cents and about 1 dB of level, while the inner conditions
are indistinguishable from the reference.

### The training pitch range is the range that works

A wide transposition sweep on one JSUT-song utterance (median f0 392 Hz) shows
where control ends, and that it ends exactly at the edge of the data. The right
column adds the four low-pitched VocalSet singers described in
[preprocessing.md](preprocessing.md#recipe-vocalset-for-the-low-pitch-range);
nothing else about the two runs differs.

| transposition | ≈ f0 | JSUT-song only | + VocalSet males |
| --- | --- | --- | --- |
| 0 | 392 Hz | 33.6 cent / corr 0.993 | 33.6 / 0.993 |
| −12st | 196 Hz | 35.7 / 0.992 | 34.2 / 0.993 |
| −14st | 174 Hz | **212.5 / 0.746** | 34.6 / 0.993 |
| −17st | 147 Hz | **337.3 / 0.633** | 42.8 / 0.989 |
| −19st | 131 Hz | **200.9 / 0.809** | 37.6 / 0.991 |
| −24st | 98 Hz | **391.4 / 0.479** | 51.6 / 0.984 |

Control collapses below about 175 Hz in the left column. JSUT-song's 1st
percentile is 225 Hz, so that is the edge of the training distribution and not
a property of the architecture: the same model trained with 45 % of its frames
below 200 Hz tracks a −24st transposition to within 52 cents.

Loudness is the weaker half. `energy_bias_db` still falls to −3.9 dB at −24st
even in the mixed run (from −5.3 dB), so deep transposition comes out quieter
than asked for. The envelope loss is computed on waveform amplitude, where a
low-f0 signal of equal RMS has a different crest factor, and nothing in the
current objective corrects for that.

Two cautions when reading these numbers on a mixed corpus:

* `val/latent_usage` is **not** comparable across corpora. VocalSet utterances
  are one sustained vowel, so permuting `z` along time removes nothing that
  should matter and the metric reads near 0 by construction. Split it by
  speaker before concluding anything: on the mixed run it is 0.245 on
  JSUT-song utterances against 0.252 for the JSUT-only run — unchanged — while
  the single-vowel speakers score 0.003–0.13 and drag the average to 0.05.
* `val/f0_rmse_cent` improves (17.8 → 9.9 cent) partly because the mixed
  validation set is dominated by sustained vowels, which are easier to track
  than sung Japanese.

## Checkpoints

`ModelCheckpoint` monitors `val/mel` and writes to `checkpoint.dirpath`, keeping
`save_top_k` best plus `last.ckpt`. Checkpoints embed the phoneme table and the
speaker map, so inference needs nothing but the `.ckpt` file.

## Multi-GPU

```bash
uv run python scripts/train.py --config configs/train/base.yml \
    trainer.devices=4 trainer.strategy=ddp
```

The bucket sampler shards across replicas by rank. Note that the batch size in
the config is **per replica**.

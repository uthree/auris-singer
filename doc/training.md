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
  kl_aux: 1.0
  feature_matching: 1.0
  envelope: 10.0
  adversarial: 1.0
```

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

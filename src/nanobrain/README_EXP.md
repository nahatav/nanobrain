# Experiment branches (G4 / A100, step matched to the H100 hour)

All branches start from `exp/base`. Each one is a small diff, gated by flags that default to off, so
with the flags off a branch trains exactly like `exp/base`. Only `src/nanobrain/**` differs from
upstream main.

## Running on Colab

`run_experiment.sh NAME ARGS...` runs `python -m nanobrain.train --overrides output_dir=output name=NAME ARGS...`,
then the open-task eval and the Drive backup. `nanobrain.train` merges
`default_pretrain.yaml < --config FILE < overrides`, where the overrides are every `--overrides` group
plus any bare `key=value` token, so this works:

```
bash /content/run_experiment.sh NAME --config src/nanobrain/config/g4_base.yaml key=value ...
```

Before the first run on a runtime (the branches must be on the fork first):

```
cd /content/nanobrain && git fetch origin exp/base && git checkout exp/base && uv sync
bash /content/fetch_data.sh /content/nanobrain/src/nanobrain/config/L2_qc_rand138.txt   # 137.8 GB
# L4 is another 137.6 GB; L2 and L4 together are 245 GB (16.1K images shared)
```

Quick smoke test without the eval (about 2 minutes plus compile):

```
cd /content/nanobrain && uv run --no-sync python -m nanobrain.train --config src/nanobrain/config/g4_base.yaml \
    output_dir=output name=smoke-NAME total_steps=300 print_freq=20 checkpoint_period=100000000
```

## exp/base

RyanKim17920's best public recipe (`ofx/ms-natinit` a039eaf, labless ms-ni-wu03m03-s7338, open 0.5521,
5-seed mean 0.5471), made runnable off the H100 and keyed to steps. Config: `config/g4_base.yaml`.

| change | flags (default) | notes |
|---|---|---|
| FA3 optional | `flash_attention_impl` (FA3), `flash_attention_fallback` (true), `sdpa_backends` (null; g4_base: flash, cudnn, efficient) | FA3 only on sm_90 and only if importable. Otherwise a warning and torch sdpa. The H100 path is unchanged. |
| step keyed schedule | `warmup_frac`, `decay_frac` (null = off) | Fractions of `total_steps`; needs `epochs: null` and `time_schedule: false`, so the step count never follows the filelist size. g4_base: 29783 steps (Ryan's H100 hour), warmup 0.08 (Ryan: 0.3 epochs of the QC list = 2376 steps), decay 0.2 (= `time_decay_frac` 0.2). |
| natinit download | `natinit.hf_repo` (facebook/vit-mae-base), `hf_revision` (25b184b), `cache_dir` (null) | Used when `natinit.path` does not exist; g4_base sets `path: null`. |
| filelists | `train_filelist` | `config/L2_qc_rand138.txt` (g4_base default, QC random control) and `config/L4_evalmatch138.txt` (eval matched). |
| FINO gate | `fino3d.exclude_collections` | Now ds001510, ds001554 and ds003974 (OpenBHB subject overlaps). L2 and L4 contain none of the three; Ryan's QC list has 208 of their images. |
| loader workers | `num_workers` (g4_base 12) | Also capped at the cpu count. |
| CLI | | Repeated `--overrides` and bare `key=value` after `--config`. |
| env | | pyproject.toml, uv.lock and scripts restored to upstream main (no flash-attn-3; brainmarks-smri pin d1eef0e, Ryan's run used 8ef7115). |

Other differences from Ryan's run config: checkpoint every 2500 steps (Ryan: only at the end), and
FINO's gamma ramp follows `step / total_steps` instead of the wall clock (the same curve at a
steady step rate).

```
bash /content/run_experiment.sh g4base-L2-s7338 --config src/nanobrain/config/g4_base.yaml
bash /content/run_experiment.sh g4base-L4-s7338 --config src/nanobrain/config/g4_base.yaml \
    train_filelist=src/nanobrain/config/L4_evalmatch138.txt
```

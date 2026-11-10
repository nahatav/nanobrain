import argparse
import datetime
import functools
import gc
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
try:
    from torch.nn.attention import activate_flash_attention_impl, current_flash_attention_impl
except ImportError:  # older torch: no pluggable flash impl, sdpa only
    activate_flash_attention_impl = current_flash_attention_impl = None
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

import nanobrain.misc as misc
import nanobrain.prep as prep
from nanobrain.data import (
    Fomo300kDataset,
    InfiniteSampler,
    brle_to_dense,
    fit_to_shape,
    native_spacing,
    normalize_values,
    reorient_to_ras,
    resample_to_spacing,
)
from nanobrain.fino import ramp as fino_ramp
from nanobrain.model import ViTMAE3D, patches_to_volume
from nanobrain.optim import CombinedOptimizer, Muon, is_muon_param
from nanobrain.visualization import plot_mask_pred

logger = logging.getLogger()

ROOT = Path(__file__).parents[2]
# brainmarks-smri data folder, shared with the eval
DATA_ROOT = Path(os.environ["BRAINMARKS_SMRI_DATA"])
DEFAULT_CONFIG = Path(__file__).parent / "config/default_pretrain.yaml"


def main(args: DictConfig):
    # setup
    process_start_time = time.monotonic()
    if args.distributed:
        # ddp under torchrun, one process per gpu. batch_size is per gpu
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl")
    rank, world_size = misc.get_rank(), misc.get_world_size()
    device = torch.device(args.device)
    # the ranks draw different views; rank 0's model init is broadcast by ddp
    misc.random_seed(args.seed + rank)

    if args.name and not args.output_dir.endswith(args.name):
        args.output_dir = f"{args.output_dir}/{args.name}"
    output_dir = Path(args.output_dir)

    if misc.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plots").mkdir(exist_ok=True)
        out_cfg_path = output_dir / "config.yaml"
        if out_cfg_path.exists():
            prev_cfg = OmegaConf.load(out_cfg_path)
            assert args == prev_cfg, "current config doesn't match previous config"
        else:
            OmegaConf.save(args, out_cfg_path)
    args.wandb = args.wandb and misc.is_main_process()

    misc.setup_logging(logger, output_dir / "log.txt")

    logger.info("pretraining nanobrain")
    logger.info(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"cwd: {Path.cwd()}")
    logger.info(f"git: {misc.git_info()}")
    if device.type == "cuda":
        logger.info(f"gpu: {torch.cuda.get_device_name(device)}")
    logger.info(f"torch: {torch.__version__} (cuda {torch.version.cuda})")
    logger.info(f"config:\n{OmegaConf.to_yaml(args)}")

    summary_path = output_dir / "summary.json"
    if summary_path.exists() and json.loads(summary_path.read_text())["done"]:
        logger.info(f"training already done, see {summary_path}")
        return

    if args.wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.name,
            notes=args.notes,
            config=OmegaConf.to_container(args),
        )

    snapshot = None
    if misc.is_main_process():
        snapshot_args = (ROOT / "src", output_dir / "source.tar.gz")
        if args.fast_startup:
            # ~2 s of gzip off the clock, joined before the summary
            snapshot = threading.Thread(target=misc.snapshot_source, args=snapshot_args)
            snapshot.start()
        else:
            misc.snapshot_source(*snapshot_args)

    # the example plots assume the 1 mm grid
    assert not (args.native and args.plot_period), "set plot_period 0 with native"

    setup_attention(args, device)

    fino_meta = None
    if args.get("fino3d") and args.fino3d.enabled:
        # metadata first: the group vocabulary sizes the heads (stored in the checkpoint args)
        from nanobrain.fino import build_meta

        assert args.native, "fino3d: native training only"
        fino_meta = build_meta(
            args.fino3d,
            Path(args.train_filelist).read_text().split(),
            DATA_ROOT / "fomo300k" / "tables" / "images.parquet",
        )

    # model
    model = ViTMAE3D.from_config(args)
    model.decoder_prune_last = args.decoder_prune_last
    model.level_batched = bool(args.get("level_batched", False))
    if args.get("natinit") and args.natinit.enabled:
        # natural-image pretrained blocks (+ inflated patch embedding). a resumed checkpoint
        # (misc.load_model below) overwrites this, as it should
        from nanobrain.natinit import apply_natural_init

        apply_natural_init(model, args.natinit)
    model.to(device)
    logger.info(f"model:\n{model}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"num params: {num_params / 1e6:.1f}M")

    if args.get("batch_flops"):
        # startup: skip the eager fwd + bwd under FlopCounterMode, take a count logged by an earlier run
        batch_flops = float(args.batch_flops)
        logger.info(f"flops per batch per gpu (fwd + bwd): {batch_flops / 1e12:.2f} TFLOP (from config)")
    else:
        batch_flops = count_batch_flops(args, model, device)
        logger.info(f"flops per batch per gpu (fwd + bwd): {batch_flops / 1e12:.2f} TFLOP")

    model_without_ddp = model
    if args.distributed:
        model = DDP(model, broadcast_buffers=False)

    compile_cache = args.get("compile_cache") if args.compile else None
    if compile_cache:
        load_compile_cache(Path(compile_cache))

    if args.compile and args.native:
        # the image shapes vary, so only the fixed-shape transformer parts compile. cudagraphs:
        # those parts (fixed shapes) replay as cuda graphs, fwd and bwd
        assert not (args.cudagraphs and args.distributed), "cudagraphs is single gpu only"
        model_without_ddp.compile_native(**({"mode": "reduce-overhead"} if args.cudagraphs else {}))
    elif args.compile and args.cudagraphs:
        assert not args.distributed, "cudagraphs is single gpu only"
        model_without_ddp.compile_cudagraphs()
    elif args.compile:
        # in place compile keeps the state dict keys
        model.compile()

    # optimizer
    # batch size counts sequences, following mae_st which scales by repeated samples
    total_batch_size = args.batch_size * world_size * args.num_samples * args.accum_iter
    logger.info(
        f"total batch size: {total_batch_size} = "
        f"{args.batch_size} images x {args.num_samples} samples x {args.accum_iter} accum "
        f"x {world_size} gpus"
    )

    if not args.get("lr"):
        args.lr = args.base_lr * total_batch_size / 256
        logger.info(f"lr: {args.lr:.2e} = {args.base_lr:.2e} x {total_batch_size} / 256")
    else:
        logger.info(f"lr: {args.lr:.2e}")

    optimizer = make_optimizer(args, model_without_ddp)

    dataset = Fomo300kDataset(DATA_ROOT / "fomo300k", args.train_filelist)
    dataset.meta = fino_meta
    logger.info(f"train dataset: {len(dataset)} images")

    if args.epochs:
        # schedule in epochs: the step counts follow from the dataset size and the global batch
        steps_per_epoch = len(dataset) / (args.batch_size * world_size * args.accum_iter)
        args.total_steps = round(args.epochs * steps_per_epoch)
        args.warmup_steps = round(args.warmup_epochs * steps_per_epoch)
        args.decay_steps = round(args.decay_epochs * steps_per_epoch)
        logger.info(f"{args.epochs} epochs of {steps_per_epoch:.1f} steps")

    if args.get("warmup_frac") is not None or args.get("decay_frac") is not None:
        # step keyed fractions of total_steps: warmup over the first warmup_frac, linear decay to
        # min_lr over the last decay_frac (null = keep warmup_steps / decay_steps). total_steps is
        # exactly what the config sets, whatever the filelist size, so a run off the H100 (time_limit
        # null, total_steps = the H100 hour's step count) gets the schedule shape of that hour
        assert not args.get("time_schedule"), "warmup_frac / decay_frac are step keyed: time_schedule false"
        assert not args.epochs, (
            "warmup_frac / decay_frac: set epochs null (epochs re-derives total_steps from the filelist size)"
        )
        if args.get("warmup_frac") is not None:
            args.warmup_steps = round(args.warmup_frac * args.total_steps)
        if args.get("decay_frac") is not None:
            args.decay_steps = round(args.decay_frac * args.total_steps)
        assert args.warmup_steps + args.decay_steps <= args.total_steps, "warmup + decay > total_steps"
        logger.info(
            f"step keyed schedule: {args.total_steps} steps = "
            f"{args.total_steps * args.batch_size * world_size * args.accum_iter / len(dataset):.2f} "
            f"epochs of {len(dataset)} images"
        )

    if args.get("time_schedule"):
        # wall-clock keyed schedule: warmup over warmup_steps, constant, then a linear decay to
        # min_lr over the last time_decay_frac of the training time, ending at time_limit. the
        # step count is open ended, so the anneal always lands on the wall whatever the step rate
        assert args.time_limit, "time_schedule needs time_limit"
        assert not args.anneal_filelist, "time_schedule: anneal_filelist switches by step count"
        args.total_steps = 10**9
        lr_schedule = None
        logger.info(
            f"lr schedule: time keyed, warmup = {args.warmup_steps} steps, decay over the last "
            f"{args.time_decay_frac:.0%} of training time, ends at {args.time_limit} min"
        )
    else:
        lr_schedule = make_lr_schedule(
            args.lr, args.min_lr, args.total_steps, args.warmup_steps, args.decay_steps
        )
        logger.info(
            f"lr schedule: total steps = {args.total_steps}, warmup = {args.warmup_steps}, "
            f"decay = {args.decay_steps}"
        )

    # loss scaling not needed for bfloat16 (according to timm)
    if args.amp and args.amp_dtype != "bfloat16":
        loss_scaler = torch.GradScaler(device.type)
    else:
        loss_scaler = None

    # load checkpoint/resume training
    start_step, prev_wall_time = misc.load_model(args, model_without_ddp, optimizer, loss_scaler)

    # data loader. one endless pass over shuffled epochs, resumed at the next unseen image
    late_indices = switch_at = None
    if args.anneal_filelist:
        # the lr decay phase draws only from this subset of train_filelist
        late = set(Path(args.anneal_filelist).read_text().split())
        late_indices = [ii for ii, path in enumerate(dataset.paths) if path in late]
        switch_at = (args.total_steps - args.decay_steps) * args.accum_iter * args.batch_size * world_size
        logger.info(f"anneal: {len(late_indices)} images from sample {switch_at}")
    # spacing_bucket: K > 0 regroups each window of K batches by native voxel spacing (fewer eager
    # per-spacing passes in forward_native, same images per window); the base stream then starts at 0
    bucket_pool = int(args.get("spacing_bucket") or 0)
    assert not bucket_pool or args.native, "spacing_bucket: native training only"
    sampler = InfiniteSampler(
        len(dataset),
        seed=args.seed,
        start=0 if bucket_pool else start_step * args.accum_iter * args.batch_size * world_size,
        rank=0 if bucket_pool else rank,
        world_size=1 if bucket_pool else world_size,
        late_indices=late_indices,
        switch_at=switch_at,
    )
    if args.get("subject_temper"):
        from nanobrain.data import subject_temper_weights

        sampler.weights = subject_temper_weights(dataset.paths, float(args.subject_temper))
        n_eff = sampler.weights.sum() ** 2 / sampler.weights.pow(2).sum()
        logger.info(
            f"subject tempered sampling: alpha {args.subject_temper}, "
            f"effective images {n_eff:.0f} of {len(dataset)}"
        )
    # never more loader workers than cpus (Colab G4: 12; the H100 reference box: 16)
    num_workers = min(args.num_workers, len(os.sched_getaffinity(0)))
    if num_workers < args.num_workers:
        logger.info(f"num_workers {args.num_workers} capped to {num_workers} cpus")
    fast_prep = not args.native
    if bucket_pool:
        from nanobrain.data import SpacingBucketBatchSampler, spacing_keys

        keys, key_spacings = spacing_keys(
            dataset.paths, DATA_ROOT / "fomo300k" / "tables" / "images.parquet", args.target_spacing
        )
        batch_sampler = SpacingBucketBatchSampler(
            sampler,
            keys,
            args.batch_size,
            pool=bucket_pool,
            seed=args.seed,
            start=start_step * args.accum_iter * world_size,
            rank=rank,
            world_size=world_size,
        )
        logger.info(
            f"spacing bucketed batches: pools of {bucket_pool} batches x {args.batch_size} images, "
            f"{len(key_spacings)} distinct spacings in the filelist"
        )
        loader_kwargs = {"batch_sampler": batch_sampler}
    else:
        loader_kwargs = {"batch_size": args.batch_size, "sampler": sampler, "drop_last": True}
    train_loader = DataLoader(
        dataset,
        **loader_kwargs,
        num_workers=num_workers,
        # no collate (variable shapes); the fast prep plans its numpy part in the workers
        collate_fn=list if not fast_prep else functools.partial(
            prep.plan_batch,
            target_spacing=args.target_spacing,
            clip_quantiles=tuple(args.clip_quantiles),
            lean=args.prep_lean,
        ),
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
        worker_init_fn=ignore_sigterm,
    )
    if fast_prep:
        batch_transform = prep.make_batch_transform(
            args.grid_size, args.target_spacing, tuple(args.clip_quantiles), lean=args.prep_lean
        )
    else:
        batch_transform = make_batch_transform(args)
    if args.prep_threads:
        train_loader = misc.prep_thread(train_loader, batch_transform, device, args.prep_threads)
        batch_transform = None

    # on SIGTERM (e.g. preemption) finish the current step, save and exit. resumes when relaunched
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_event.set())

    stats = train(
        args,
        model,
        model_without_ddp,
        train_loader,
        batch_transform,
        optimizer,
        loss_scaler,
        lr_schedule,
        start_step,
        prev_wall_time,
        process_start_time,
        device,
        batch_flops,
        stop_event,
    )

    if compile_cache and misc.is_main_process():
        save_compile_cache(Path(compile_cache))
    if args.distributed:
        dist.destroy_process_group()
    if snapshot is not None:
        snapshot.join()
    if not misc.is_main_process():
        return

    step = stats["step"]
    images = step * args.accum_iter * args.batch_size * world_size
    summary = {
        "step": step,
        "images": images,
        "epochs": images / len(dataset),
        "train_flops": step * args.accum_iter * batch_flops * world_size,
        "wall_seconds": stats["wall_time"],
        "start_step": start_step,
        "done": stats["done"],
        "stopped_by_signal": stop_event.is_set(),
        "final_loss": stats["loss"],
        "num_params": num_params,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "num_gpus": world_size,
        "cpus": len(os.sched_getaffinity(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git": misc.git_info(),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"summary:\n{json.dumps(summary, indent=2)}")
    wall_time_str = datetime.timedelta(seconds=int(stats["wall_time"]))
    if stats["done"]:
        logger.info(f"done! step {step}, wall time: {wall_time_str}")
    else:
        logger.info(f"stopped by signal at step {step}. wall time: {wall_time_str}")


def train(
    args: DictConfig,
    model: nn.Module,
    model_without_ddp: nn.Module,
    data_loader: Iterable,
    batch_transform: Callable | None,
    optimizer: torch.optim.Optimizer,
    loss_scaler: torch.GradScaler | None,
    lr_schedule: np.ndarray,
    start_step: int,
    prev_wall_time: float,
    start_time: float,
    device: torch.device,
    batch_flops: float,
    stop_event: threading.Event,
) -> dict:
    model.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("grad", misc.SmoothedValue())
    metric_logger.add_meter("loss", misc.SmoothedValue())
    metric_logger.add_meter(
        "vol/s", misc.SmoothedValue(window_size=1, fmt="{value:.0f} ({global_avg:.0f})")
    )
    metric_logger.add_meter(
        "tflop/s", misc.SmoothedValue(window_size=1, fmt="{value:.0f} ({global_avg:.0f})")
    )
    metric_logger.add_meter(
        "watts", misc.SmoothedValue(window_size=1, fmt="{value:.0f} ({global_avg:.0f})")
    )

    # vol/s counts all gpus, tflop/s is per gpu
    throughput = misc.ThroughputMeter(
        {"vol": args.batch_size * misc.get_world_size(), "tflop": batch_flops / 1e12},
        device=device,
    )

    # steps count optimizer updates, batches count loader batches
    total_steps = args.total_steps if not args.debug else start_step + 10
    print_freq = args.print_freq if not args.debug else 1
    print_freq_batches = print_freq * args.accum_iter
    num_batches = total_steps * args.accum_iter
    start_batch = start_step * args.accum_iter

    amp_dtype = getattr(torch, args.amp_dtype)
    output_dir = Path(args.output_dir)
    log_path = output_dir / "log.json"
    # time limit on wall time from the start of main, summed over restarts
    time_limit = args.time_limit * 60 if args.time_limit else float("inf")
    # H100 guard: the leaderboard budget is 1 h on one H100. A step-keyed run (total_steps matched to
    # an H100 hour from off-H100 timing) also stops at h100_time_limit minutes when it runs on an H100,
    # so a maintainer rerun cannot exceed the budget even if the step estimate was optimistic
    h100_time_limit = args.get("h100_time_limit")
    if h100_time_limit and device.type == "cuda" and "H100" in torch.cuda.get_device_name(device):
        time_limit = min(time_limit, h100_time_limit * 60)
        logger.info(f"H100 guard: training stops at {h100_time_limit} min wall time")
    # full-context cooldown: the last ctx_cooldown.frac of training sees ctx_cooldown.num_visible
    # visible tokens, closer to the ~3500 tokens per image the eval encodes
    ctx_cooldown = args.get("ctx_cooldown") or {}
    cooldown_frac = float(ctx_cooldown.get("frac") or 0.0)
    cooldown_logged = False
    step = start_step
    done = False
    time_schedule = bool(args.get("time_schedule"))
    # training clock origin of the time schedule: wall time at the end of the first step (after
    # compile and loader warmup). wall time is summed over restarts (checkpoint wall_time), so a
    # resumed (requeued / preempted) run must keep the ORIGINAL origin, persisted in the output
    # dir; re-keying to the resume point would restart the anneal. without the file (older runs),
    # fall back to the first-step origin of a fresh run (~70 s), capped at the resume point
    t0_path = output_dir / "time_schedule_t0.json"
    train_t0 = None
    if start_step > 0:
        if t0_path.exists():
            train_t0 = float(json.loads(t0_path.read_text())["train_t0"])
        else:
            train_t0 = min(prev_wall_time, 70.0)
        logger.info(
            f"resumed at step {start_step}, wall {prev_wall_time:.0f} s: time schedule origin "
            f"{train_t0:.0f} s"
        )

    optimizer.zero_grad()

    if args.high_priority_stream and device.type == "cuda":
        # model work on a high priority stream: the prep threads' kernels (priority 0 side streams)
        # only take the sms it leaves free. autograd runs the backward on the forward's stream
        stream = torch.cuda.Stream(device, priority=-1)
        stream.wait_stream(torch.cuda.current_stream(device))
        torch.cuda.set_stream(stream)

    batches = metric_logger.log_every(
        data_loader,
        print_freq_batches,
        "Train:",
        total_steps=num_batches,
        start_step=start_batch,
    )
    for batch_idx, batch in enumerate(batches, start=start_batch):
        need_update = (batch_idx + 1) % args.accum_iter == 0
        log_step = (batch_idx + 1) % print_freq_batches == 0
        if time_schedule:
            lr = time_keyed_lr(
                args, step, prev_wall_time + time.monotonic() - start_time, time_limit, train_t0
            )
        else:
            lr = float(lr_schedule[step])
        if need_update:
            misc.update_lr(optimizer.param_groups, lr)

        # run volume processing on gpu, not cpu workers, ~25x faster (prep threads: already done)
        if batch_transform is not None:
            batch = batch_transform(misc.send_data(batch, device))
        images = batch["image"]
        masks = batch["mask"]
        if batch.get("spacing") is not None:
            # distinct spacings = eager per-spacing passes in forward_native (spacing_bucket lowers it)
            metric_logger.update(groups=len(set(map(tuple, batch["spacing"]))))

        if args.cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        fino_kwargs = {}
        if "fino" in batch:
            # FINO's gamma ramp over training progress (the time schedule's clock when time keyed)
            if time_schedule:
                wall = prev_wall_time + time.monotonic() - start_time
                progress = 0.0 if train_t0 is None else (wall - train_t0) / max(time_limit - train_t0, 1.0)
            else:
                progress = step / total_steps
            fino_kwargs = {
                "fino_meta": batch["fino"],
                "fino_gamma": fino_ramp(args.fino3d.gamma_max, progress),
            }
        num_visible = args.num_visible
        if cooldown_frac:
            if time_schedule:
                wall = prev_wall_time + time.monotonic() - start_time
                cool_progress = 0.0 if train_t0 is None else (wall - train_t0) / max(time_limit - train_t0, 1.0)
            else:
                cool_progress = step / total_steps
            if cool_progress >= 1.0 - cooldown_frac:
                num_visible = int(ctx_cooldown["num_visible"])
                if not cooldown_logged:
                    logger.info(f"ctx cooldown: num_visible {args.num_visible} -> {num_visible} at step {step}")
                    cooldown_logged = True
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            loss = model(
                images,
                masks,
                num_visible=num_visible,
                num_predict=args.num_predict,
                num_samples=args.num_samples,
                with_state=False,
                spacing=batch.get("spacing"),
                **fino_kwargs,
            )

        # the cuda graph's output buffer is overwritten by the next replay
        loss_for_log = loss.detach().clone() if args.cudagraphs else loss.detach()
        torch._assert_async(torch.isfinite(loss_for_log), "non-finite loss")

        grad_norm = misc.backward_step(
            loss / args.accum_iter,
            optimizer,
            scaler=loss_scaler,
            need_update=need_update,
            max_norm=args.clip_grad,
        )

        # log tensors without syncing, values are only read on log steps
        metric_logger.update(loss=loss_for_log)
        if need_update:
            metric_logger.update(lr=lr, grad=grad_norm)

        throughput.step()
        if log_step:
            metric_logger.update(**throughput.compute())
            log_stats = {
                "step": step + 1,
                "loss": metric_logger.loss.value,
                "lr": lr,
                "grad": metric_logger.grad.value,
                "vol_per_sec": metric_logger.meters["vol/s"].value,
                "tflop_per_sec": metric_logger.meters["tflop/s"].value,
                "watts": metric_logger.meters["watts"].value,
            }
            if "groups" in metric_logger.meters:
                log_stats["spacing_groups"] = metric_logger.meters["groups"].global_avg
            if fino_kwargs:
                log_stats["fino_gamma"] = fino_kwargs["fino_gamma"]
                for k, v in model_without_ddp.fino_stats.items():
                    log_stats[f"fino_{k}"] = float(v)
            for k, v in model_without_ddp.level_mirror_stats.items():
                log_stats[f"mirror_{k}"] = float(v)
            if misc.is_main_process():
                with log_path.open("a") as f:
                    print(json.dumps(log_stats), file=f)
            if args.wandb:
                wandb.log({f"train/{k}": v for k, v in log_stats.items()}, step=step + 1)

        if not need_update:
            continue
        step += 1
        if args.gc_freeze and step == start_step + 1:
            # compile is done: later gen 2 collections skip the startup objects
            gc.collect()
            gc.freeze()
        wall_time = prev_wall_time + time.monotonic() - start_time
        if train_t0 is None:
            train_t0 = wall_time
            if misc.is_main_process():
                t0_path.write_text(json.dumps({"train_t0": train_t0, "step": step}))
        done = step == total_steps or wall_time >= time_limit
        is_last = done or stop_event.is_set()
        if args.distributed:
            # every rank stops with rank 0, else the others hang in the next all-reduce
            flags = torch.tensor([done, is_last], dtype=torch.int32, device=device)
            dist.broadcast(flags, src=0)
            done, is_last = (bool(f) for f in flags.tolist())

        # plot a few images from the current batch
        num_examples = args.plot_nrow * args.plot_ncol
        # not on stop by signal, just save and exit
        is_plot_step = args.plot_period and (step % args.plot_period == 0 or done)
        if num_examples and is_plot_step and misc.is_main_process():
            plot_paths = plot_examples(
                args, model_without_ddp, images, masks, step, amp_dtype, device
            )
            if args.wandb:
                wandb.log(
                    {f"train/{name}": wandb.Image(str(path)) for name, path in plot_paths.items()},
                    step=step,
                )

        is_save_step = step % args.checkpoint_period == 0 or step in (args.checkpoint_steps or [])
        if is_save_step or is_last:
            misc.save_model(args, step, model_without_ddp, optimizer, loss_scaler, wall_time)

        if is_last:
            break

    logger.info(f"Averaged stats: {metric_logger}")
    return {
        "step": step,
        "done": done,
        "wall_time": prev_wall_time + time.monotonic() - start_time,
        "loss": metric_logger.loss.median,
    }


def make_optimizer(args: DictConfig, model: nn.Module):
    # cast or else it corrupts the checkpoint
    betas = tuple(args.betas) if args.betas is not None else None
    if args.optimizer == "adamw":
        param_groups = misc.get_param_groups(model)
        misc.update_lr(param_groups, args.lr)
        misc.update_wd(param_groups, args.weight_decay)
        return torch.optim.AdamW(param_groups, betas=betas, fused=args.fused_adamw)
    assert args.optimizer == "muon", f"unknown optimizer {args.optimizer}"
    # muon for the block matrices, adamw for the rest (embeddings, head, norms, biases)
    muon_names = [
        n
        for n, p in model.named_parameters()
        if is_muon_param(n, p, args.muon_encoder_only, args.muon_decoder_only, args.muon_all_matrices)
    ]
    logger.info(f"muon params: {len(muon_names)} matrices")
    # natinit.block_lr_mult: lr multiplier of the pretrained encoder blocks + final norm.
    # natinit.layer_decay: layer-wise lr decay on top (block i of depth d gets
    # block_lr_mult * layer_decay ** (d - 1 - i); the final norm keeps block_lr_mult). 1.0 = off
    block_mult, layer_decay, depth = 1.0, 1.0, len(model.blocks)
    if args.get("natinit") and args.natinit.enabled:
        block_mult = float(args.natinit.block_lr_mult)
        layer_decay = float(args.natinit.get("layer_decay", 1.0))
    from nanobrain.natinit import is_natinit_param

    def lr_mult_fn(name):
        if not is_natinit_param(name):
            return 1.0
        if name.startswith("blocks."):
            return block_mult * layer_decay ** (depth - 1 - int(name.split(".")[1]))
        return block_mult

    adamw_groups = misc.get_param_groups(model, skip_names=set(muon_names), lr_mult_fn=lr_mult_fn)
    params = dict(model.named_parameters())
    muon_groups = []
    for mult in sorted({lr_mult_fn(n) for n in muon_names}):
        muon_groups.append(
            {
                "params": [params[n] for n in muon_names if lr_mult_fn(n) == mult],
                "lr_multiplier": args.muon_lr_mult * mult,
                "wd_multiplier": 1.0,
            }
        )
    if block_mult != 1.0 or layer_decay != 1.0:
        logger.info(
            f"natinit block_lr_mult {block_mult} layer_decay {layer_decay}: muon groups "
            f"{[(len(g['params']), g['lr_multiplier']) for g in muon_groups]}, adamw groups "
            f"{[(len(g['params']), g['lr_multiplier']) for g in adamw_groups]}"
        )
    for groups in (adamw_groups, muon_groups):
        misc.update_lr(groups, args.lr)
        misc.update_wd(groups, args.weight_decay)
    return CombinedOptimizer(
        [
            torch.optim.AdamW(adamw_groups, betas=betas, fused=args.fused_adamw),
            Muon(muon_groups, momentum=args.muon_momentum, fused=args.muon_fused),
        ]
    )


SDPA_BACKENDS = {
    "flash": torch.backends.cuda.enable_flash_sdp,
    "cudnn": torch.backends.cuda.enable_cudnn_sdp,
    "efficient": torch.backends.cuda.enable_mem_efficient_sdp,
    "math": torch.backends.cuda.enable_math_sdp,
}


def setup_attention(args: DictConfig, device: torch.device) -> str:
    # flash_attention_impl (e.g. FA3 = flash-attn-3 behind sdpa) with sdpa's cudnn / mem efficient
    # backends off so flash runs: the H100 path, unchanged. FA3 is Hopper (sm_90) only, so on any
    # other gpu (G4 Blackwell sm_120, A100 sm_80) or without the package the run falls back to
    # torch's own sdpa kernels (same attention math, different kernel) unless
    # flash_attention_fallback is false. sdpa_backends (null = torch's choice) restricts the
    # fallback to a subset of flash / cudnn / efficient / math. Returns the attention path used
    impl = args.get("flash_attention_impl")
    if impl and device.type == "cuda":
        cap = torch.cuda.get_device_capability(device)
        reason = None
        if activate_flash_attention_impl is None:
            reason = "this torch has no activate_flash_attention_impl"
        elif str(impl).upper() == "FA3" and cap != (9, 0):
            reason = f"FA3 needs sm_90, this gpu is sm_{cap[0]}{cap[1]}"
        else:
            try:
                activate_flash_attention_impl(impl)
            except Exception as e:  # noqa: BLE001
                reason = f"{type(e).__name__}: {e}"
        if reason is None:
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            logger.info(f"flash attention impl: {current_flash_attention_impl()}")
            return str(impl)
        if not args.get("flash_attention_fallback", True):
            raise RuntimeError(f"flash_attention_impl {impl} unavailable: {reason}")
        logger.warning(f"flash_attention_impl {impl} unavailable ({reason}); using torch sdpa")
    backends = args.get("sdpa_backends")
    if backends and device.type == "cuda":
        unknown = set(backends) - set(SDPA_BACKENDS)
        assert not unknown, f"sdpa_backends: unknown {sorted(unknown)}, use {sorted(SDPA_BACKENDS)}"
        for name, enable in SDPA_BACKENDS.items():
            enable(name in backends)
    logger.info(f"attention: torch sdpa, backends {list(backends) if backends else 'torch default'}")
    return "sdpa"


def load_compile_cache(path: Path) -> None:
    # Inductor / AOTAutograd mega-cache (torch.compiler.save_cache_artifacts) from an earlier run of the
    # same code, torch and gpu: compile becomes cache hits (no Triton codegen or autotune). A stale or
    # foreign cache only costs the normal compile
    if not path.exists():
        logger.info(f"compile cache: {path} not found, compiling from scratch (saved at the end)")
        return
    try:
        info = torch.compiler.load_cache_artifacts(path.read_bytes())
        logger.info(f"compile cache: loaded {path} ({path.stat().st_size / 1e6:.1f} MB): {info}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"compile cache: load of {path} failed ({e!r}), compiling from scratch")


def save_compile_cache(path: Path) -> None:
    # after training (off the step clock), only when there is no cache yet: also holds the graphs the
    # ctx cooldown recompiles for its longer sequences
    if path.exists():
        return
    try:
        out = torch.compiler.save_cache_artifacts()
        if out is None:
            logger.info("compile cache: nothing to save")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(out[0])
        logger.info(f"compile cache: saved {path} ({len(out[0]) / 1e6:.1f} MB)")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"compile cache: save to {path} failed ({e!r})")


def ignore_sigterm(worker_id: int):
    # `timeout` signals the whole process group. workers keep loading until the main process exits
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def time_keyed_lr(
    args: DictConfig, step: int, wall_time: float, time_limit: float, train_t0: float | None
) -> float:
    lr, min_lr = args.lr, args.min_lr
    if step < args.warmup_steps:
        return lr * step / args.warmup_steps
    if train_t0 is None:
        return lr
    progress = (wall_time - train_t0) / max(time_limit - train_t0, 1.0)
    decay_start = 1.0 - args.time_decay_frac
    if progress <= decay_start:
        return lr
    frac = min((progress - decay_start) / args.time_decay_frac, 1.0)
    return lr + (min_lr - lr) * frac


def make_lr_schedule(
    lr: float, min_lr: float, total_steps: int, warmup_steps: int, decay_steps: int
) -> np.ndarray:
    warmup = np.linspace(0.0, lr, warmup_steps, endpoint=False)
    stable = np.full(total_steps - warmup_steps - decay_steps, lr)
    decay = np.linspace(lr, min_lr, decay_steps)
    return np.concatenate([warmup, stable, decay])


def make_batch_transform(args: DictConfig):
    def fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        raw = batch
        batch = [
            sample_transform(
                sample,
                target_spacing=args.target_spacing,
                grid_size=tuple(args.grid_size),
                clip_quantiles=tuple(args.clip_quantiles),
                native=args.native,
            )
            for sample in batch
        ]
        if args.native:
            # each image on its own voxel grid
            out = {key: [sample[key] for sample in batch] for key in ("image", "mask", "spacing")}
            if "fino" in raw[0]:
                out["fino"] = torch.from_numpy(np.stack([sample["fino"] for sample in raw]))
            return out
        images = torch.stack([sample["image"] for sample in batch])
        masks = torch.stack([sample["mask"] for sample in batch])
        return {"image": images, "mask": masks}

    return fn


def sample_transform(
    sample: dict[str, Tensor],
    *,
    target_spacing: float = 1.0,
    grid_size: tuple[int, int, int] = (192, 240, 192),
    clip_quantiles: tuple[float, float] = (0.005, 0.995),
    native: bool = False,
) -> dict[str, Tensor]:
    # pretraining preprocessing, on gpu. eval preprocessing is ViTMAE3D.transform_cuda, keep in sync
    # (native training still evaluates on 1 mm inputs)
    values = sample["values"].float() * sample["scale"]
    values = normalize_values(values, clip_quantiles)
    mask = brle_to_dense(sample["mask_rle"], shape=sample["shape"])
    image = values.new_zeros(mask.shape).masked_scatter_(mask, values)
    image, mask, affine = reorient_to_ras(image, mask, sample["affine"])
    if native:
        # no resampling. forward_native fits the grid_size mm box in the image's own voxels
        spacing = native_spacing(affine, target_spacing)
        return {"image": image, "mask": mask, "affine": affine, "spacing": spacing}
    image, mask, affine = resample_to_spacing(image, mask, affine, target_spacing)
    image, mask, affine = fit_to_shape(image, mask, affine, grid_size)
    return {"image": image, "mask": mask, "affine": affine}


def count_batch_flops(args: DictConfig, model: nn.Module, device: torch.device) -> float:
    # fwd + bwd flops of one training batch, on dummy images where every patch is in the mask
    amp_dtype = getattr(torch, args.amp_dtype)
    images = torch.zeros(args.batch_size, *args.grid_size, device=device)
    masks = torch.ones(args.batch_size, *args.grid_size, device=device)
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            loss = model(
                images,
                masks,
                num_visible=args.num_visible,
                num_predict=args.num_predict,
                num_samples=args.num_samples,
                with_state=False,
            )
        loss.backward()
    model.zero_grad(set_to_none=True)
    return flop_counter.get_total_flops()


def plot_examples(
    args: DictConfig,
    model: nn.Module,
    images: Tensor,
    masks: Tensor,
    step: int,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Path]:
    num_examples = args.plot_nrow * args.plot_ncol
    images = images[:num_examples]
    masks = masks[:num_examples]

    model.eval()
    # eager, the plot shapes differ from training and compiling them isn't worth it
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp),
        torch.compiler.set_stance("force_eager"),
    ):
        _, state = model(
            images,
            masks,
            num_visible=args.num_visible,
            num_predict=args.plot_num_predict,
            num_samples=1,
        )
    model.train()

    grid_size = tuple(args.grid_size)
    patch_size = args.patch_size
    vis_ids = state["vis_ids"]
    target_ids = state["target_ids"]
    pred = patches_to_volume(state["pred_patches"].float(), target_ids, grid_size, patch_size)
    visible_mask = patches_to_volume(
        torch.ones_like(state["vis_patches"]), vis_ids, grid_size, patch_size
    )
    pred_mask = patches_to_volume(
        torch.ones_like(state["target_patches"]), target_ids, grid_size, patch_size
    )

    images = images.float().cpu().numpy()
    masks = masks.cpu().numpy()
    pred = pred.cpu().numpy()
    visible_mask = visible_mask.cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()

    plot_dir = Path(args.output_dir) / "plots"
    plot_paths = {}
    for view in ["sag", "cor", "ax"]:
        fig = plot_mask_pred(
            images,
            pred,
            visible_mask,
            pred_mask=pred_mask,
            img_mask=masks,
            view=view,
            nrow=args.plot_nrow,
            ncol=args.plot_ncol,
        )
        path = plot_dir / f"mae_{view}_{step:07d}.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths[f"mae_{view}"] = path
    return plot_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    # --overrides may repeat, and bare key=value tokens after --config also count as overrides, so
    # `--overrides name=x --config file.yaml total_steps=N` works (default < config < overrides)
    parser.add_argument("--overrides", type=str, default=None, nargs="+", action="extend")
    args, rest = parser.parse_known_args()
    bad = [tok for tok in rest if tok.startswith("-") or "=" not in tok]
    if bad:
        parser.error(f"unrecognized arguments: {' '.join(bad)}")
    overrides = (args.overrides or []) + rest
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.config:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.config))
    if overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(overrides))
    main(cfg)

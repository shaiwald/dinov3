# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from functools import partial
import logging
import numpy as np
import os
import random
import wandb

import torch
import torch.distributed as dist

from dinov3.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader, make_dataset
import dinov3.distributed as distributed
from dinov3.eval.segmentation.eval import evaluate_segmentation_model, is_main_process, colorize_mask
from dinov3.eval.segmentation.loss import MultiSegmentationLoss
from dinov3.eval.segmentation.metrics import SEGMENTATION_METRICS
from dinov3.eval.segmentation.models import build_segmentation_decoder
from dinov3.eval.segmentation.schedulers import build_scheduler
from dinov3.eval.segmentation.transforms import make_segmentation_eval_transforms, make_segmentation_train_transforms
from dinov3.logging import MetricLogger, SmoothedValue

logger = logging.getLogger("dinov3")


# Standard Cityscapes 19-class palette
CITYSCAPES_PALETTE = np.asarray([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)


class InfiniteDataloader:
    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader
        self.data_iterator = iter(dataloader)
        self.sampler = dataloader.sampler
        if not hasattr(self.sampler, "epoch"):
            self.sampler.epoch = 0  # type: ignore

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self.dataloader)

    def __next__(self):
        try:
            data = next(self.data_iterator)
        except StopIteration:
            self.sampler.epoch += 1
            self.data_iterator = iter(self.dataloader)
            data = next(self.data_iterator)
        return data


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.
    The seed of each worker equals to num_worker * rank + worker_id + user_seed
    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def validate(
    segmentation_model: torch.nn.Module,
    val_dataloader,
    device,
    autocast_dtype,
    eval_res,
    eval_stride,
    decoder_head_type,
    num_classes,
    global_step,
    metric_to_save,
    current_best_metric_to_save_value,
    reduce_zero_label,
):
    new_metric_values_dict = evaluate_segmentation_model(
        segmentation_model,
        val_dataloader,
        device,
        eval_res,
        eval_stride,
        decoder_head_type,
        num_classes,
        autocast_dtype,
        reduce_zero_label,
    )
    logger.info(f"Step {global_step}: {new_metric_values_dict}")
    # Log Metrics to WandB
    if is_main_process():
        wandb.log({f"Val/{k}": v for k, v in new_metric_values_dict.items()})

    # `segmentation_model` is a module list of [backbone, decoder]
    # Only put the head in train mode
    segmentation_model.module.segmentation_model[1].train()
    is_better = False
    if new_metric_values_dict[metric_to_save] > current_best_metric_to_save_value:
        is_better = True
    return is_better, new_metric_values_dict


def train_step(
    segmentation_model: torch.nn.Module,
    batch,
    device,
    scaler,
    optimizer,
    optimizer_gradient_clip,
    scheduler,
    criterion,
    model_dtype,
    global_step,
):
    # a) load batch
    batch_img, (_, gt) = batch
    batch_img = batch_img.to(device)  # B x C x h x w
    gt = gt.to(device)  # B x (num_classes if multilabel) x h x w
    optimizer.zero_grad(set_to_none=True)

    # b) forward pass
    with torch.autocast("cuda", dtype=model_dtype, enabled=True if model_dtype is not None else False):
        pred = segmentation_model(batch_img)  # B x num_classes x h x w
        gt = torch.squeeze(gt).long()  # Adapt gt dimension to enable loss calculation

    # c) compute loss
    if gt.shape[-2:] != pred.shape[-2:]:
        pred = torch.nn.functional.interpolate(input=pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    loss = criterion(pred, gt)

    # --- WANDB VISUALIZATION START (Every 500 steps) ---
    if is_main_process() and global_step % 500 == 0:
        with torch.no_grad():
            # Get the first image in the batch
            img_vis = batch_img[0].detach().float().cpu().permute(1, 2, 0).numpy()
            # Denormalize (Approximate for ImageNet stats: mean=[0.485...], std=[0.229...])
            img_vis = (img_vis * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            img_vis = np.clip(img_vis * 255, 0, 255).astype(np.uint8)

            # Process Prediction (Argmax) & Ground Truth
            pred_mask = torch.argmax(pred[0], dim=0).detach().cpu().numpy()
            gt_mask = gt[0].detach().cpu().numpy()

            wandb.log({
                "Train/Example": wandb.Image(
                    img_vis,
                    caption=f"Step {global_step}",
                    masks={
                        "predictions": {"mask_data": pred_mask, "class_labels": {i: f"{i}" for i in range(19)}},
                        "ground_truth": {"mask_data": gt_mask, "class_labels": {i: f"{i}" for i in range(19)}}
                    }
                ),
                # Log a colored composite for easier quick viewing
                "Train/Composite": wandb.Image(
                    np.concatenate([img_vis, colorize_mask(gt_mask), colorize_mask(pred_mask)], axis=1),
                    caption="Img | GT | Pred"
                )
            })
    # --- WANDB VISUALIZATION END ---

    # d) optimization
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(segmentation_model.module.parameters(), optimizer_gradient_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(segmentation_model.module.parameters(), optimizer_gradient_clip)
        optimizer.step()

    if global_step > 0:  # inheritance from old mmcv code
        scheduler.step()

    return loss


def train_segmentation(
    backbone,
    config,
):
    assert config.decoder_head.type == "linear", "Only linear head is supported for training"
    # 1- load the segmentation decoder
    logger.info("Initializing the segmentation model")
    segmentation_model = build_segmentation_decoder(
        backbone,
        config.decoder_head.backbone_out_layers,
        "linear",
        num_classes=config.decoder_head.num_classes,
        autocast_dtype=config.model_dtype.autocast_dtype,
        dropout=config.decoder_head.dropout,
    )
    global_device = distributed.get_rank()
    local_device = torch.cuda.current_device()
    segmentation_model = torch.nn.parallel.DistributedDataParallel(
        segmentation_model.to(local_device), device_ids=[local_device]
    )  # should be local rank
    model_parameters = filter(lambda p: p.requires_grad, segmentation_model.parameters())
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model_parameters)}")

    # 2- create data transforms + dataloaders
    train_transforms = make_segmentation_train_transforms(
        img_size=config.transforms.train.img_size,
        random_img_size_ratio_range=config.transforms.train.random_img_size_ratio_range,
        crop_size=config.transforms.train.crop_size,
        flip_prob=config.transforms.train.flip_prob,
        reduce_zero_label=config.eval.reduce_zero_label,
        mean=config.transforms.mean,
        std=config.transforms.std,
    )
    val_transforms = make_segmentation_eval_transforms(
        img_size=config.transforms.eval.img_size,
        inference_mode=config.eval.mode,
        mean=config.transforms.mean,
        std=config.transforms.std,
    )

    train_dataset = DatasetWithEnumeratedTargets(
        make_dataset(
            dataset_str=f"{config.datasets.train}:root={config.datasets.root}",
            transforms=train_transforms,
        )
    )
    train_sampler_type = None
    if distributed.is_enabled():
        train_sampler_type = SamplerType.DISTRIBUTED
    init_fn = partial(
        worker_init_fn, num_workers=config.num_workers, rank=global_device, seed=config.seed + global_device
    )
    train_dataloader = InfiniteDataloader(
        make_data_loader(
            dataset=train_dataset,
            batch_size=config.bs,
            num_workers=config.num_workers,
            sampler_type=train_sampler_type,
            shuffle=True,
            persistent_workers=False,
            worker_init_fn=init_fn,
        )
    )

    val_dataset = DatasetWithEnumeratedTargets(
        make_dataset(
            dataset_str=f"{config.datasets.val}:root={config.datasets.root}",
            transforms=val_transforms,
        )
    )
    val_sampler_type = None
    if distributed.is_enabled():
        val_sampler_type = SamplerType.DISTRIBUTED
    val_dataloader = make_data_loader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=config.num_workers,
        sampler_type=val_sampler_type,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
    )

    # 3- define and create scaler, optimizer, scheduler, loss
    scaler = None
    if config.model_dtype.autocast_dtype is not None:
        scaler = torch.amp.GradScaler("cuda")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": filter(lambda p: p.requires_grad, segmentation_model.parameters()),
                "lr": config.optimizer.lr,
                "betas": (config.optimizer.beta1, config.optimizer.beta2),
                "weight_decay": config.optimizer.weight_decay,
            }
        ]
    )
    scheduler = build_scheduler(
        config.scheduler.type,
        optimizer=optimizer,
        lr=config.optimizer.lr,
        total_iter=config.scheduler.total_iter,
        constructor_kwargs=config.scheduler.constructor_kwargs,
    )
    criterion = MultiSegmentationLoss(
        diceloss_weight=config.train.diceloss_weight, celoss_weight=config.train.celoss_weight
    )
    total_iter = config.scheduler.total_iter
    global_step = 0
    global_best_metric_values = {metric: 0.0 for metric in SEGMENTATION_METRICS}

    # 5- train the model
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", SmoothedValue(window_size=4, fmt="{value:.3f}"))
    for batch in metric_logger.log_every(
        train_dataloader,
        50,
        header="Train: ",
        start_iteration=global_step,
        n_iterations=total_iter,
    ):
        if global_step >= total_iter:
            break
        loss = train_step(
            segmentation_model,
            batch,
            local_device,
            scaler,
            optimizer,
            config.optimizer.gradient_clip,
            scheduler,
            criterion,
            config.model_dtype.autocast_dtype,
            global_step,
        )

        if is_main_process() and global_step % 10 == 0:
            wandb.log({
                "Train/Loss": loss.item(),
                "Train/LR": optimizer.param_groups[0]["lr"],
                "Train/Step": global_step
            })
        # ----------------------
        global_step += 1
        metric_logger.update(loss=loss)
        if global_step % config.eval.eval_interval == 0:
            dist.barrier()
            is_better, best_metric_values_dict = validate(
                segmentation_model,
                val_dataloader,
                local_device,
                config.model_dtype.autocast_dtype,
                config.eval.crop_size,
                config.eval.stride,
                config.decoder_head.type,
                config.decoder_head.num_classes,
                global_step,
                config.metric_to_save,
                global_best_metric_values[config.metric_to_save],
                config.eval.reduce_zero_label,
            )
            if is_better:
                logger.info(f"New best metrics at Step {global_step}: {best_metric_values_dict}")
                global_best_metric_values = best_metric_values_dict

        # one last validation only if the number of total iterations is NOT divisible by eval interval:
        if total_iter % config.eval.eval_interval:
            is_better, best_metric_values_dict = validate(
                segmentation_model,
                val_dataloader,
                local_device,
                config.model_dtype.autocast_dtype,
                config.eval.crop_size,
                config.eval.stride,
                config.decoder_head.type,
                config.decoder_head.num_classes,
                global_step,
                config.metric_to_save,
                global_best_metric_values[config.metric_to_save],
                config.eval.reduce_zero_label,
            )
            if is_better:
                logger.info(f"New best metrics at Step {global_step}: {best_metric_values_dict}")
                global_best_metric_values = best_metric_values_dict
    logger.info("Training is done!")
    # segmentation_model is a module list of [backbone, decoder]
    # Only save the decoder head
    torch.save(
        {
            "model": {k: v for k, v in segmentation_model.module.state_dict().items() if "segmentation_model.1" in k},
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(config.output_dir, "model_final.pth"),
    )
    logger.info(f"Final best metrics: {global_best_metric_values}")
    return global_best_metric_values

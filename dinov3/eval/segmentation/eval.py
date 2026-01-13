# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from functools import partial
import logging

import torch

import dinov3.distributed as distributed
from dinov3.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader, make_dataset
from dinov3.eval.segmentation.inference import make_inference
from dinov3.eval.segmentation.metrics import (
    calculate_intersect_and_union,
    calculate_segmentation_metrics,
)
from dinov3.eval.segmentation.models import build_segmentation_decoder
from dinov3.eval.segmentation.transforms import make_segmentation_eval_transforms
from dinov3.hub.segmentors import dinov3_vit7b16_ms
from dinov3.logging import MetricLogger
import wandb
import numpy as np

# Cityscapes Palette
CITYSCAPES_PALETTE = np.asarray([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

def colorize_mask(mask):
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(CITYSCAPES_PALETTE)):
        color_img[mask == class_id] = CITYSCAPES_PALETTE[class_id]
    return color_img

def is_main_process():
    return not distributed.is_enabled() or distributed.get_rank() == 0

logger = logging.getLogger("dinov3")

RESULTS_FILENAME = "results-semantic-segmentation.csv"
MAIN_METRICS = ["mIoU"]


def evaluate_segmentation_model(
    segmentation_model: torch.nn.Module,
    test_dataloader,
    device,
    eval_res,
    eval_stride,
    decoder_head_type,
    num_classes,
    autocast_dtype,
    reduce_zero_label,
    wandb_vis_batch_limit=0, #visualization of only the First Batch (i == 0)
):
    segmentation_model = segmentation_model.to(device)
    segmentation_model.eval()
    all_metric_values = []
    metric_logger = MetricLogger(delimiter="  ")

    for i, (batch_img, (_, gt)) in enumerate(metric_logger.log_every(test_dataloader, 10, header="Validation: ")):
        batch_img = [img.to(device).to(dtype=autocast_dtype) for img in batch_img]
        gt = gt.to(device)[0]
        aggregated_preds = torch.zeros(1, num_classes, gt.shape[-2], gt.shape[-1])
        for img_idx, img in enumerate(batch_img):
            aggregated_preds += make_inference(
                img,
                segmentation_model.module,
                inference_mode="slide",
                decoder_head_type=decoder_head_type,
                rescale_to=gt.shape[-2:],
                n_output_channels=num_classes,
                crop_size=eval_res if isinstance(eval_res, (tuple, list)) else (eval_res, eval_res),
                stride=eval_stride if isinstance(eval_stride, (tuple, list)) else (eval_stride, eval_stride),
                apply_horizontal_flip=(img_idx and img_idx >= len(batch_img) / 2),
                output_activation=partial(torch.nn.functional.softmax, dim=1),
            )
        aggregated_preds = (aggregated_preds / len(batch_img)).argmax(dim=1, keepdim=True).to(device)

        # =========================================================
        # === WANDB VISUALIZATION ===
        # =========================================================
        wandb_should_log = (wandb_vis_batch_limit is None) or (i <= wandb_vis_batch_limit)
        if is_main_process() and wandb_should_log and wandb.run is not None:
            try:
                # 1. Prepare Image
                # batch_img[0] is the CPU tensor list. batch_img[0][0] is the first scale.
                img_vis = batch_img[0][0].float().detach().cpu().permute(1, 2, 0).numpy()
                img_vis = (img_vis * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                img_vis = np.clip(img_vis * 255, 0, 255).astype(np.uint8)

                # 2. Prepare GT & Preds
                # .squeeze() ensures we remove singleton dimensions (1, H, W) -> (H, W)
                # preventing the "too many values to unpack" error
                gt_vis = gt.detach().cpu().squeeze().numpy()
                pred_vis = aggregated_preds.detach().cpu().squeeze().numpy()

                # 3. Setup Naming
                if wandb_vis_batch_limit is None:
                    # Inference Mode: Unique key for every batch
                    log_key = "Test/inference_all"
                    caption = f"Inference_Image_{i}: Input | GT | Pred"
                else:
                    # Training Mode: Fixed key, overwrites itself
                    log_key = "Val/training_preview"
                    caption = f"Preview_Batch_{i}: Input | GT | Pred"

                # 4. Log
                wandb.log({
                    log_key: [wandb.Image(
                        np.concatenate([
                            img_vis,
                            colorize_mask(gt_vis),
                            colorize_mask(pred_vis)
                        ], axis=1),
                        caption=caption
                    )]
                })
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")
        # =========================================================

        intersect_and_union = calculate_intersect_and_union(
            aggregated_preds[0],
            gt,
            num_classes=num_classes,
            reduce_zero_label=reduce_zero_label,
        )
        all_metric_values.append(intersect_and_union)
        del img, gt, aggregated_preds, intersect_and_union

    all_metric_values = torch.stack(all_metric_values)
    if distributed.is_enabled():
        all_metric_values = torch.cat(distributed.gather_all_tensors((all_metric_values)))
    final_metrics = calculate_segmentation_metrics(
        all_metric_values,
        metrics=["mIoU", "dice", "fscore"],
    )
    final_metrics = {k: round(v.cpu().item() * 100, 2) for k, v in final_metrics.items()}
    logger.info(final_metrics)
    return final_metrics


def test_segmentation(backbone, config):
    # 1- construct a segmentation decoder
    if config.load_from == "dinov3_vit7b16_ms":  # torch hub descriptor
        # Load public m2f head checkpoints
        logger.info("Loading the 7B backbone and the M2F adapter with torchhub")
        segmentation_model = dinov3_vit7b16_ms(autocast_dtype=config.model_dtype.autocast_dtype, check_hash=True)
    else:
        segmentation_model = build_segmentation_decoder(
            backbone,
            config.decoder_head.backbone_out_layers,
            config.decoder_head.type,
            hidden_dim=config.decoder_head.hidden_dim,  # Only used for instantiating a M2F head
            num_classes=config.decoder_head.num_classes,
            autocast_dtype=config.model_dtype.autocast_dtype,
            dropout=config.decoder_head.dropout,
        )
        state_dict = torch.load(config.load_from, map_location="cpu")["model"]
        _, _ = segmentation_model.load_state_dict(state_dict, strict=False)
    device = distributed.get_rank()
    segmentation_model = torch.nn.parallel.DistributedDataParallel(segmentation_model.to(device), device_ids=[device])

    # 2- dataloader for testing
    eval_res = config.eval.crop_size
    eval_stride = config.eval.stride
    transforms = make_segmentation_eval_transforms(
        img_size=eval_res,
        inference_mode="slide",
        use_tta=config.eval.use_tta,
        tta_ratios=config.transforms.eval.tta_ratios,
        mean=config.transforms.mean,
        std=config.transforms.std,
    )

    test_dataset = DatasetWithEnumeratedTargets(
        make_dataset(
            dataset_str=f"{config.datasets.val}:root={config.datasets.root}",
            transforms=transforms,
        )
    )

    test_sampler_type = None
    if distributed.is_enabled():
        test_sampler_type = SamplerType.DISTRIBUTED

    test_dataloader = make_data_loader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=6,
        sampler_type=test_sampler_type,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
    )

    # 3- make inference
    metric_values_dict = evaluate_segmentation_model(
        segmentation_model=segmentation_model,
        test_dataloader=test_dataloader,
        device=device,
        eval_res=eval_res,
        eval_stride=eval_stride,
        decoder_head_type=config.decoder_head.type,
        num_classes=config.decoder_head.num_classes,
        autocast_dtype=config.model_dtype.autocast_dtype,
        reduce_zero_label=config.eval.reduce_zero_label,
        wandb_vis_batch_limit=None,
    )

    # Log Metrics to WandB
    if is_main_process():
        wandb.log({f"Test/{k}": v for k, v in metric_values_dict.items()})

    return metric_values_dict

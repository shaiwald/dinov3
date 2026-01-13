# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
import logging
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

from .decoders import Decoder, DenseTargetDecoder, ImageDataDecoder
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    TRAIN_REF = "train_ref"
    VAL_REF = "val_ref"
    TEST_REF = "test_ref"

    @property
    def dirname(self) -> str:
        return self.value


def _load_acdc_paths(root: str, split: _Split, condition: Optional[str] = None) -> Tuple[List[str], List[str]]:
    image_paths = []
    target_paths = []

    # Valid ACDC conditions
    all_conditions = ["fog", "night", "rain", "snow"]

    # Filter conditions if user requested a specific one
    if condition is not None:
        if condition not in all_conditions:
            raise ValueError(f"Invalid ACDC condition '{condition}'. Must be one of {all_conditions}")
        conditions_to_load = [condition]
    else:
        conditions_to_load = all_conditions

    # Default folders based on README
    img_type_folder = "rgb_anon"
    gt_type_folder = "gt"
    split_name = split.dirname

    for cond in conditions_to_load:
        # Path: {root}/rgb_anon/{condition}/{split}
        img_folder = os.path.join(root, img_type_folder, cond, split_name)
        tgt_folder = os.path.join(root, gt_type_folder, cond, split_name)

        if not os.path.exists(img_folder):
            logger.warning(f"ACDC condition folder not found: {img_folder}. Skipping.")
            continue

        # Walk through sequences (e.g., GOPR0351)
        for seq_name in sorted(os.listdir(img_folder)):
            seq_img_dir = os.path.join(img_folder, seq_name)
            seq_tgt_dir = os.path.join(tgt_folder, seq_name)

            if not os.path.isdir(seq_img_dir):
                continue

            for file_name in sorted(os.listdir(seq_img_dir)):

                # Construct Ground Truth Filename
                # Robust replacement for both anon and non-anon images
                if "_rgb_anon.png" in file_name:
                    target_name = file_name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
                elif "_rgb.png" in file_name:
                    target_name = file_name.replace("_rgb.png", "_gt_labelTrainIds.png")
                elif "_rgb_ref_anon.png" in file_name:
                    target_name = file_name.replace("_rgb_ref_anon.png", "_gt_ref_labelTrainIds.png")
                elif "_rgb_ref.png" in file_name:
                    target_name = file_name.replace("_rgb_ref.png", "_gt_ref_labelTrainIds.png")
                else:
                    continue

                full_tgt_path = os.path.join(seq_tgt_dir, target_name)
                full_img_path = os.path.join(seq_img_dir, file_name)

                # Append paths if GT exists (or if it's a test split where GT might be hidden)
                if os.path.exists(full_tgt_path):
                    image_paths.append(os.path.relpath(full_img_path, root))
                    target_paths.append(os.path.relpath(full_tgt_path, root))
                elif split in [_Split.TEST, _Split.TEST_REF]:
                    # Allow loading test images without GT
                    image_paths.append(os.path.relpath(full_img_path, root))
                    # Check if your pipeline handles missing targets (e.g. via None or skipping in __getitem__)
                    # For strict compatibility, we only append if we have a target or if the pipeline handles it.
                    pass

    logger.info(f"Loaded {len(image_paths)} ACDC samples for split '{split_name}' (Conditions: {conditions_to_load})")
    return image_paths, target_paths


class ACDC(ExtendedVisionDataset):
    Split = _Split
    Labels = Union[Image.Image]

    def __init__(
            self,
            split: Union[_Split, str],
            root: Optional[str] = None,
            condition: Optional[str] = None,  # <--- Added argument
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            image_decoder: Decoder = ImageDataDecoder,
            target_decoder: Decoder = DenseTargetDecoder,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=image_decoder,
            target_decoder=target_decoder,
        )

        if isinstance(split, str):
            try:
                split = _Split[split]
            except KeyError:
                split = _Split(split.lower())

        self.image_paths, self.target_paths = _load_acdc_paths(root, split, condition)

    def get_image_data(self, index: int) -> bytes:
        image_relpath = self.image_paths[index]
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Any:
        if index >= len(self.target_paths):
            raise ValueError("Target not found. This might be a test split without annotations.")

        target_relpath = self.target_paths[index]
        target_full_path = os.path.join(self.root, target_relpath)
        with open(target_full_path, mode="rb") as f:
            target_data = f.read()
        return target_data

    def __len__(self) -> int:
        return len(self.image_paths)
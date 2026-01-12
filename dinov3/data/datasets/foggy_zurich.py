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
    TESTV1 = "testv1"
    TESTV2 = "testv2"
    LIGHT = "light"
    MEDIUM = "medium"

    @property
    def dirname(self) -> str:
        return self.value


def _load_foggy_zurich_paths(root: str, split: _Split) -> Tuple[List[str], List[str]]:
    # Map standard splits to Foggy Zurich specific splits
    if split == _Split.VAL:
        fz_split = "testv2"
    elif split == _Split.TEST:
        fz_split = "testv2"
    elif split == _Split.TRAIN:
        fz_split = "medium"
    else:
        fz_split = split.value

    path_list_dir = os.path.join(root, "lists_file_names")
    if not os.path.exists(path_list_dir):
        raise FileNotFoundError(f"Could not find lists directory: {path_list_dir}")

    # File names: RGB_testv2_filenames.txt, gt_labelTrainIds_testv2_filenames.txt
    rgb_list_filename = f"RGB_{fz_split}_filenames.txt"
    tgt_list_filename = f"gt_labelTrainIds_{fz_split}_filenames.txt"

    rgb_list_path = os.path.join(path_list_dir, rgb_list_filename)
    tgt_list_path = os.path.join(path_list_dir, tgt_list_filename)

    # 1. Load RGB Paths
    if not os.path.exists(rgb_list_path):
        raise FileNotFoundError(f"Split list not found: {rgb_list_path}")

    with open(rgb_list_path, "r") as f:
        # The text files ALREADY include the 'RGB/' prefix
        rgb_lines = [line.strip() for line in f.readlines() if line.strip()]

    # 2. Load Target Paths (only if they exist)
    tgt_lines = []
    has_targets = os.path.exists(tgt_list_path)

    if has_targets:
        with open(tgt_list_path, "r") as f:
            # The text files ALREADY include the 'gt_labelTrainIds/' prefix
            tgt_lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(rgb_lines) != len(tgt_lines):
            logger.warning(
                f"Mismatch in FoggyZurich split {fz_split}: "
                f"{len(rgb_lines)} images vs {len(tgt_lines)} targets."
            )

    image_paths = []
    target_paths = []

    for i, rel_path in enumerate(rgb_lines):
        # Path in text file: RGB/GP0.../img.png
        # We use it directly.
        image_paths.append(rel_path)

        if has_targets and i < len(tgt_lines):
            # Path in text file: gt_labelTrainIds/GP0.../img.png
            target_paths.append(tgt_lines[i])

    logger.info(f"Loaded {len(image_paths)} images for split {fz_split}")
    return image_paths, target_paths


class FoggyZurich(ExtendedVisionDataset):
    Split = _Split
    Labels = Union[Image.Image]

    def __init__(
            self,
            split: Union[_Split, str],
            root: Optional[str] = None,
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

        self.image_paths, self.target_paths = _load_foggy_zurich_paths(root, split)

    def get_image_data(self, index: int) -> bytes:
        image_relpath = self.image_paths[index]
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Any:
        if not self.target_paths:
            raise ValueError("No ground truth loaded for this split.")

        target_relpath = self.target_paths[index]
        target_full_path = os.path.join(self.root, target_relpath)
        with open(target_full_path, mode="rb") as f:
            target_data = f.read()
        return target_data

    def __len__(self) -> int:
        return len(self.image_paths)
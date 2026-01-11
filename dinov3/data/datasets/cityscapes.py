# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

from .decoders import Decoder, DenseTargetDecoder, ImageDataDecoder
from .extended import ExtendedVisionDataset


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"

    @property
    def dirname(self) -> str:
        return {
            _Split.TRAIN: "train",
            _Split.VAL: "val",
        }[self]


def _load_cityscapes_paths(root: str, split: _Split) -> Tuple[List[str], List[str]]:
    image_paths = []
    target_paths = []

    # Use the Enum's .dirname property (e.g. "train" or "val")
    img_root = os.path.join(root, "leftImg8bit", split.dirname)
    tgt_root = os.path.join(root, "gtFine", split.dirname)

    if not os.path.exists(img_root):
        raise FileNotFoundError(f"Cityscapes images not found at: {img_root}")

    # CONFIG: Which fog intensity to use?
    # Options usually: "beta_0.005", "beta_0.01", "beta_0.02"
    # If you want ALL of them, set this to None (but this duplicates validation scenes!)
    TARGET_BETA = "beta_0.02"

    # Walk through the city folders (e.g., aachen, bochum)
    for city in sorted(os.listdir(img_root)):
        city_img_dir = os.path.join(img_root, city)
        city_tgt_dir = os.path.join(tgt_root, city)

        if not os.path.isdir(city_img_dir):
            continue

        for file_name in sorted(os.listdir(city_img_dir)):
            # 1. Check if it's a valid image file
            if not file_name.endswith(".png"):
                continue

            # 2. Filter for specific Fog Intensity (if using Foggy Cityscapes)
            if "foggy" in file_name and TARGET_BETA and (TARGET_BETA not in file_name):
                continue

            # 3. Determine the unique ID of the image (e.g., frankfurt_000000_000294)
            # Standard: frankfurt_000000_000294_leftImg8bit.png
            # Foggy:    frankfurt_000000_000294_leftImg8bit_foggy_beta_0.01.png
            if "_leftImg8bit" in file_name:
                unique_id = file_name.split("_leftImg8bit")[0]
            else:
                continue  # Skip if naming convention is totally unknown

            # 4. Construct the Ground Truth filename
            # The GT is ALWAYS standard (never foggy), e.g., frankfurt_..._gtFine_labelTrainIds.png
            target_name = f"{unique_id}_gtFine_labelTrainIds.png"
            full_tgt_path = os.path.join(city_tgt_dir, target_name)

            if os.path.exists(full_tgt_path):
                # Store relative paths
                full_img_path = os.path.join(city_img_dir, file_name)
                image_paths.append(os.path.relpath(full_img_path, root))
                target_paths.append(os.path.relpath(full_tgt_path, root))

    return image_paths, target_paths


class Cityscapes(ExtendedVisionDataset):
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

        self.image_paths, self.target_paths = _load_cityscapes_paths(root, split)

    def get_image_data(self, index: int) -> bytes:
        image_relpath = self.image_paths[index]
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Any:
        target_relpath = self.target_paths[index]
        target_full_path = os.path.join(self.root, target_relpath)
        with open(target_full_path, mode="rb") as f:
            target_data = f.read()
        return target_data

    def __len__(self) -> int:
        return len(self.image_paths)
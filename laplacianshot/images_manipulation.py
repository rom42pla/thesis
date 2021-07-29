from copy import deepcopy
from typing import Optional

import numpy as np

import torch
from torchvision import transforms as T

from detectron2.structures import ImageList


def normalize_image(img: torch.Tensor, model: torch.nn.Module, resize: bool = True):
    # normalizes pixel's intensities
    img = model.normalizer(img)
    img = ImageList.from_tensors(
        [img], model.backbone.size_divisibility
    )[0]
    if resize:
        # resizes the image to make sides divisible by a factor
        sides_lengths = []
        for dimension in [1, 2]:
            if img.shape[dimension] % model.backbone.size_divisibility == 0:
                sides_lengths += [img.shape[dimension]]
            else:
                sides_lengths += [img.shape[dimension] +
                                  (model.backbone.size_divisibility -
                                   img.shape[dimension] % model.backbone.size_divisibility)]
        img = T.Resize(sides_lengths)(img)
    return img


def apply_random_augmentation(img: torch.Tensor,
                              box: Optional[torch.Tensor] = None):
    transformations = T.Compose(
        [
            T.ColorJitter(brightness=0.3, saturation=0.2, hue=0.1),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.01, 1)),
            T.RandomAutocontrast(p=0.1),
            # T.RandomEqualize(p=0.1)
        ]
    )
    img_augmented, box_augmented = transformations(img), deepcopy(box)

    # random horizontal flip
    if np.random.random() <= 0.5:
        img_augmented = T.RandomHorizontalFlip(p=1)(img_augmented)

        box_augmented[0], box_augmented[2] = img_augmented.shape[2] - box_augmented[2], \
                                             img_augmented.shape[2] - box_augmented[0]

    # random vertical flip
    if np.random.random() <= 0.1:
        img_augmented = T.RandomVerticalFlip(p=1)(img_augmented)

        box_augmented[1], box_augmented[3] = img_augmented.shape[1] - box_augmented[3], \
                                             img_augmented.shape[1] - box_augmented[1]

    # random crop
    box_augmented[0] = box_augmented[0] + (box_augmented[2] - box_augmented[0]) * 0.4 * np.random.random()
    box_augmented[2] = box_augmented[2] - (box_augmented[2] - box_augmented[0]) * 0.4 * np.random.random()
    box_augmented[1] = box_augmented[1] + (box_augmented[3] - box_augmented[1]) * 0.4 * np.random.random()
    box_augmented[3] = box_augmented[3] - (box_augmented[3] - box_augmented[1]) * 0.4 * np.random.random()

    if box is None:
        return img_augmented
    else:
        return img_augmented, box_augmented

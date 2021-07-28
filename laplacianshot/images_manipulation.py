import torch

from detectron2.structures import ImageList
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2


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


def apply_random_augmentation(img: torch.Tensor):
    transformations = T.Compose(
        [
            T.ColorJitter(brightness=0.3, saturation=0.2, hue=0.1),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.01, 5)),
            T.RandomAutocontrast(p=0.1),
            T.RandomEqualize(p=0.1)
        ]
    )
    img_augmented = transformations(img)
    # img_augmented = transformations(image=img.cpu().numpy())["image"] \
    #     .permute(1, 2, 0).to(img.device)
    return img_augmented

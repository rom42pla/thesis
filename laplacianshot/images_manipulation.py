import torch

from detectron2.structures import ImageList
from torchvision import transforms as T


def normalize_image(img: torch.Tensor, model: torch.nn.Module):
    # normalizes pixel's intensities
    img = model.normalizer(img)
    img = ImageList.from_tensors(
        [img], model.backbone.size_divisibility
    )[0]
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

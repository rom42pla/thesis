import os
import time
from os.path import join, exists
from typing import List, Union, Optional

import numpy as np
import pandas as pd

import torch

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def plot_scatterplot(embeddings_s: torch.Tensor, labels_s: torch.Tensor,
                     embeddings_q=None,
                     title: str = None, folder: str = "."):
    if not title:
        title = str(int(time.time()))
    if not exists(folder):
        os.makedirs(folder)

    pca = PCA(n_components=2)
    pca.fit(embeddings_s)

    df = pd.DataFrame([{
        "x": x,
        "y": y,
        "label": label.item()
    } for (x, y), label in zip(pca.transform(embeddings_s), labels_s)])

    if embeddings_q is not None:
        df = df.append(pd.DataFrame([{
            "x": x,
            "y": y,
            "label": "query"
        } for (x, y) in pca.transform(embeddings_q)]))

    fig, ax = plt.subplots(1, figsize=[10, 10])
    sns.scatterplot(x="x", y="y", hue="label", data=df,
                    palette="Paired", ax=ax).set_title(f"Scatterplot {title.lower()}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.savefig(join(folder, f"scatterplot_{title.lower().replace(' ', '_')}.png"))
    plt.close(fig)


def plot_prototypes_difference(prototypes: torch.Tensor, prototypes_rectified: torch.Tensor,
                               labels: torch.Tensor,
                               folder: str = "."):
    if not exists(folder):
        os.makedirs(folder)

    pca = PCA(n_components=2)
    pca.fit(torch.cat([prototypes, prototypes_rectified], dim=0))

    df_prototypes = pd.DataFrame([{
        "x": x,
        "y": y,
        "label": label.item()
    } for (x, y), label in zip(pca.transform(prototypes), labels)])

    df_prototypes_rectified = pd.DataFrame([{
        "x": x,
        "y": y,
        "label": label.item()
    } for (x, y), label in zip(pca.transform(prototypes_rectified), labels)])

    fig, (ax1, ax2) = plt.subplots(2, figsize=[10, 20])
    sns.scatterplot(x="x", y="y", hue="label", data=df_prototypes,
                    palette="Paired", ax=ax1).set_title(f"Prototypes as mean of supports")
    sns.scatterplot(x="x", y="y", hue="label", data=df_prototypes_rectified,
                    palette="Paired", ax=ax2).set_title(f"Prototypes rectified")
    for ax in (ax1, ax2):
        ax.set_xlabel("")
        ax.set_ylabel("")
    plt.savefig(join(folder, f"scatterplot_prototypes_differences.png"))
    plt.close(fig)


def plot_distribution(distribution: Union[list, torch.Tensor],
                      bins: Optional[int] = None,
                      label_x: str = "", label_y: str = "count",
                      title: str = None, folder: str = "."):
    if isinstance(distribution, list):
        distribution = torch.as_tensor(distribution)
    if not title:
        title = str(int(time.time()))
    if not exists(folder):
        os.makedirs(folder)
    if not bins:
        bins = len(distribution.unique())

    # plots the results
    fig, ax = plt.subplots(1, figsize=[10, 10])
    sns.histplot(x=distribution, bins=bins, ax=ax, palette="rocket").set_title(f"Barplot {title.lower()}")
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    plt.savefig(join(folder, f"barplot_{title.lower().replace(' ', '_')}.png"))
    plt.close(fig)


def plot_detections(img: torch.Tensor,
                    boxes: torch.Tensor, confidences: torch.Tensor, labels: torch.Tensor,
                    folder: str = "."):
    if not exists(folder):
        os.makedirs(folder)

    # sets up the layout
    fig = plt.figure(constrained_layout=True, figsize=(2 * 10,
                                                       2 * (4 + 10)))
    gs = fig.add_gridspec(4 + 10,
                          10)

    # plots the query image
    img = (img / 255).float()
    img = img[[2, 1, 0], :, :]
    ax_query_image = fig.add_subplot(gs[:4, 3:7])
    ax_query_image.set_title("query image")
    ax_query_image.imshow(img.permute(1, 2, 0))

    # loops over smaller boxes
    x, y = 0, 4
    for i_box, (box, confidence, label) in enumerate(zip(boxes, confidences, labels)):
        if i_box >= 10 * 10:
            break
        # plots a detection
        ax_box = fig.add_subplot(gs[y, x])
        ax_box.set_title(f"label={label}\nconfidence={np.round(confidence.item() * 100, 2)}%")
        img_crop = img[:,
                   int(box[1]): int(box[3]),
                   int(box[0]): int(box[2])]
        ax_box.imshow(img_crop.permute(1, 2, 0))

        # updates the cursor
        x += 1
        if x >= 10:
            x, y = 0, y + 1

    # saves the plot
    plt.title(f"Example of query detections")
    plt.savefig(join(folder, f"query_image_example.png"))
    plt.close(fig)


def plot_supports(imgs: List[torch.Tensor],
                  labels: torch.Tensor,
                  folder: str = "."):
    if not exists(folder):
        os.makedirs(folder)

    # sets up the layout
    fig = plt.figure(constrained_layout=True, figsize=(2 * int(np.ceil(len(imgs) / len(labels.unique()))),
                                                       2 * len(labels.unique()),))
    gs = fig.add_gridspec(len(labels.unique()),
                          int(np.ceil(len(imgs) / len(labels.unique()))))
    # plots the query image
    for i_label, label in enumerate(labels.unique()):
        class_indices = (labels == label).nonzero().flatten().tolist()
        for i_img, img in enumerate([img for i, img in enumerate(imgs)
                                     if i in class_indices]):
            img = (img / 255).float()
            img = img[[2, 1, 0], :, :]
            ax = fig.add_subplot(gs[i_label, i_img])
            ax.set_title(f"label {label.item()}")
            ax.imshow(img.permute(1, 2, 0))

    # saves the plot
    plt.title(f"Support samples")
    plt.savefig(join(folder, f"supports_example.png"))
    plt.close(fig)


def plot_supports_augmentations(imgs: List[torch.Tensor],
                                labels: torch.Tensor,
                                original_images_indices: List[int],
                                folder: str = "."):
    if not exists(folder):
        os.makedirs(folder)

    # sets up the layout
    fig = plt.figure(constrained_layout=True, figsize=(len(imgs) // len(original_images_indices),
                                                       len(original_images_indices),))
    gs = fig.add_gridspec(len(original_images_indices),
                          len(imgs) // len(original_images_indices))
    # plots the query image
    for i_row, i_img_original in enumerate(original_images_indices):
        if i_row < len(original_images_indices) - 1:
            imgs_in_row = imgs[i_img_original:original_images_indices[i_row+1]]
        else:
            imgs_in_row = imgs[i_img_original:]
        for i_img, img in enumerate(imgs_in_row):
            img = (img / 255).float()
            img = img[[2, 1, 0], :, :]
            ax = fig.add_subplot(gs[i_row, i_img])
            ax.set_title(f"shape {img.shape[2]} x {img.shape[1]}")
            ax.imshow(img.permute(1, 2, 0))

    # saves the plot
    plt.title(f"Support samples")
    plt.savefig(join(folder, f"supports_augmented_example.png"))
    plt.close(fig)

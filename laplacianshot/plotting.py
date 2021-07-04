import os
import time
from os.path import join, exists
import pandas as pd

import torch
from sklearn.decomposition import PCA
from pprint import pprint
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


def plot_distribution(distribution: torch.Tensor,
                      title: str = None, folder: str = "."):
    if not title:
        title = str(int(time.time()))
    if not exists(folder):
        os.makedirs(folder)

    # plots the results
    fig, ax = plt.subplots(1, figsize=[10, 10])
    sns.histplot(x=distribution, bins=20, ax=ax).set_title(f"Scatterplot {title.lower()}")
    ax.set_xlabel("")
    plt.savefig(join(folder, f"barplot_{title.lower().replace(' ', '_')}.png"))

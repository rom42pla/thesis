import itertools
from os.path import join
from typing import Optional, List

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
from sklearn.naive_bayes import GaussianNB

from tqdm import tqdm

from laplacianshot import train_lshot
from laplacianshot.plotting import plot_scatterplot, plot_distribution, plot_prototypes_difference, \
    plot_explained_variance
from laplacianshot.prototypes import get_prototypes_rectified, get_prototypes


def laplacian_shot(X_s_embeddings: torch.Tensor, X_s_labels: torch.Tensor,
                   X_q_embeddings: torch.Tensor, X_q_labels_pred: torch.Tensor,
                   X_q_pred_confidence: torch.Tensor, X_q_probabilties: torch.Tensor,
                   X_q_ids: Optional[torch.Tensor] = None,
                   embeddings_are_probabilities: bool = False,
                   null_label: int = 20,
                   proto_rect: bool = True,
                   leverage_classification: bool = True,
                   remove_possibly_duplicates: bool = False,
                   do_pca: bool = False,
                   knn: int = 3, lambda_factor: float = 0.1,
                   plots: bool = False,
                   logs: bool = True) -> torch.Tensor:
    # assures inputs are in the correct shape
    assert len(X_s_embeddings.shape) == 2, f"X_s_embeddings must have shape (n_samples, embeddings_size) " \
                                           f"but got {X_s_embeddings.shape}"
    assert len(X_s_labels.shape) == 1, f"X_s_labels must have shape (n_samples) " \
                                       f"but got {X_s_labels.shape}"
    assert len(X_q_embeddings.shape) == 2, f"X_q_embeddings must have shape (n_samples, embeddings_size) " \
                                           f"but got {X_q_embeddings.shape}"
    assert X_s_embeddings.shape[0] == X_s_labels.shape[0], f"Inconsistent number of " \
                                                           f"X_s embeddings {X_s_embeddings.shape[0]} " \
                                                           f"and labels {X_s_labels.shape[0]}"
    assert X_s_embeddings.shape[1] == X_q_embeddings.shape[1], f"Inconsistent embeddings' shape " \
                                                               f"between X_s {X_s_embeddings.shape[1]} " \
                                                               f"and X_q {X_q_embeddings.shape[1]}"
    assert not knn or isinstance(knn, int) and knn >= 1, f"knn parameter must be an integer >= 1"
    assert not lambda_factor or lambda_factor >= 0, f"lambda_factor must be None or >= 0 " \
                                                    f"but got {lambda_factor}"
    assert isinstance(proto_rect, bool)
    assert isinstance(embeddings_are_probabilities, bool)

    if plots:
        plot_scatterplot(embeddings_s=X_s_embeddings, labels_s=X_s_labels,
                         embeddings_q=X_q_embeddings,
                         title=f"before normalization", folder=join(".", "plots"))
    if plots:
        plot_distribution(distribution=X_q_pred_confidence,
                          title="Confidence of prediction", folder=join(".", "plots"))

    if logs:
        print(f"X_s_embeddings ranges in [{X_s_embeddings.float().min()}, {X_s_embeddings.float().max()}] "
              f"with mean norm {np.linalg.norm(X_s_embeddings.float(), 2, 1).mean()} before normalization")
        print(f"X_q_embeddings ranges in [{X_q_embeddings.float().min()}, {X_q_embeddings.float().max()}] "
              f"with mean norm {np.linalg.norm(X_q_embeddings.float(), 2, 1).mean()} before normalization")

    # normalizes the vectors
    if not embeddings_are_probabilities:
        mean = torch.mean(X_s_embeddings, dim=0)
        X_s_embeddings = X_s_embeddings - mean
        X_s_embeddings = X_s_embeddings / np.linalg.norm(X_s_embeddings, 2, 1)[:, None]

        X_q_embeddings = X_q_embeddings - mean
        X_q_embeddings = X_q_embeddings / np.linalg.norm(X_q_embeddings, 2, 1)[:, None]

    if logs:
        print(f"X_s_embeddings ranges in [{X_s_embeddings.float().min()}, {X_s_embeddings.float().max()}] "
              f"with mean norm {np.linalg.norm(X_s_embeddings.float(), 2, 1).mean()} before normalization")
        print(f"X_q_embeddings ranges in [{X_q_embeddings.float().min()}, {X_q_embeddings.float().max()}] "
              f"with mean norm {np.linalg.norm(X_q_embeddings.float(), 2, 1).mean()} before normalization")

    n_queries = len(X_q_embeddings)

    if plots:
        plot_scatterplot(embeddings_s=X_s_embeddings, labels_s=X_s_labels,
                         embeddings_q=X_q_embeddings,
                         title=f"after normalization", folder=join(".", "plots"))

    if proto_rect:
        prototypes = get_prototypes_rectified(embeddings_s=X_s_embeddings, labels_s=X_s_labels,
                                              embeddings_q=X_q_embeddings, pseudolabels_q=X_q_labels_pred,
                                              confidence_q=X_q_pred_confidence, add_shifting_term=True)
    else:
        prototypes = get_prototypes(embeddings=X_s_embeddings, labels=X_s_labels)

    if plots:
        plot_scatterplot(embeddings_s=prototypes, labels_s=X_s_labels.unique(),
                         embeddings_q=X_q_embeddings,
                         title=f"of prototypes and queries", folder=join(".", "plots"))

    # gets predictions with null label
    X_q_null_indices = (X_q_labels_pred == null_label).nonzero().flatten()
    X_q_non_null_indices = (X_q_labels_pred != null_label).nonzero().flatten()

    # discards prediction with null label
    X_q_embeddings = X_q_embeddings[X_q_non_null_indices]
    X_q_labels_pred = X_q_labels_pred[X_q_non_null_indices]
    X_q_pred_confidence = X_q_pred_confidence[X_q_non_null_indices]
    X_q_probabilties = X_q_probabilties[X_q_non_null_indices]
    X_q_ids = X_q_ids[X_q_non_null_indices]

    distances_values_per_id, similar_detections_indices = [], []
    if remove_possibly_duplicates:
        for img_id in X_q_ids.unique():
            detection_indices = (X_q_ids == img_id).nonzero().flatten()

            # computes distances between detections
            same_image_distances = torch.cdist(X_q_embeddings[detection_indices],
                                               X_q_embeddings[detection_indices], p=2).numpy()

            # retrieves indices of similar detections
            same_image_distances_tmp = same_image_distances.copy()
            same_image_distances_tmp[same_image_distances_tmp <= 0.1] = np.inf
            same_image_distances_tmp = np.triu(same_image_distances_tmp)
            np.fill_diagonal(same_image_distances_tmp, 0)
            same_detections_indices_image = np.argwhere(same_image_distances_tmp == np.inf)
            for v in np.unique(same_detections_indices_image.flatten()):
                same_detections_indices_image[same_detections_indices_image == v] = detection_indices[v]
            similar_detections_indices += same_detections_indices_image.tolist()

            # removes the diagonal
            distances_values_per_id += np.concatenate(
                [same_image_distances[i][same_image_distances[i] != same_image_distances[i][i]]
                 for i in range(len(same_image_distances))]).tolist()

        if plots:
            plot_distribution(distribution=distances_values_per_id, bins=100,
                              label_x="L2 distance",
                              title="Distances between same-image detections", folder=join(".", "plots"))

        graph = np.zeros(shape=(len(X_q_embeddings), len(X_q_embeddings)), dtype=np.uint8)
        for id1, id2 in similar_detections_indices:
            if X_q_pred_confidence[id1] > X_q_pred_confidence[id2]:
                graph[id1, id2] = 1
            else:
                graph[id2, id1] = 1
        duplicates_indices = np.asarray([y for x, y in np.argwhere(graph == 1)])
        non_duplicate_indices = np.delete(np.arange(len(X_q_embeddings)), duplicates_indices)


    # leverages fc classification results
    if leverage_classification:
        for i_embedding, (embedding, label, score, probabilities) in enumerate(zip(X_q_embeddings,
                                                                                   X_q_labels_pred,
                                                                                   X_q_pred_confidence,
                                                                                   X_q_probabilties)):
            # for i_prototype, prototype in enumerate(prototypes):
            #     X_q_embeddings[i_embedding] = (1 - probabilities[i_prototype]) * embedding + \
            #                                   probabilities[i_prototype] * prototype

            X_q_embeddings[i_embedding] = (1 - score) * embedding + \
                                          score * prototypes[label]

    if plots:
        plot_scatterplot(embeddings_s=prototypes, labels_s=X_s_labels.unique(),
                         embeddings_q=X_q_embeddings,
                         title=f"of prototypes and queries after leverage of induction",
                         folder=join(".", "plots"))

    if plots:
        plot_prototypes_difference(prototypes=get_prototypes(embeddings=X_s_embeddings, labels=X_s_labels),
                                   prototypes_rectified=get_prototypes_rectified(
                                       embeddings_s=X_s_embeddings, labels_s=X_s_labels,
                                       embeddings_q=X_q_embeddings, pseudolabels_q=X_q_labels_pred,
                                       confidence_q=X_q_pred_confidence, add_shifting_term=True),
                                   labels=X_s_labels.unique(),
                                   folder=join(".", "plots"))

    if do_pca and not embeddings_are_probabilities:
        # reduces dimensionality
        pca = PCA(n_components=16)
        pca.fit(X_q_embeddings.numpy())

        X_q_embeddings, prototypes = torch.from_numpy(pca.transform(X_q_embeddings)), \
                                     torch.from_numpy(pca.transform(prototypes))

        if plots:
            plot_explained_variance(variances=pca.explained_variance_ratio_.tolist(),
                                    title="Explained variance of PCA", folder=join(".", "plots"))

    if remove_possibly_duplicates:
        # removes possibly duplicates
        X_q_embeddings = X_q_embeddings[non_duplicate_indices]

    # predicts the labels
    if logs:
        print(f"Predicting {len(X_q_embeddings)} labels with Laplacianshot")
    if not embeddings_are_probabilities:
        # distance = np.linalg.norm(prototypes.numpy()[:, None, :] - X_q_embeddings.numpy(), 2, axis=-1)
        distance = torch.cdist(prototypes, X_q_embeddings, p=2).numpy()
    else:
        distance = np.zeros(shape=(len(prototypes), len(X_q_embeddings)), dtype=float)
        for i_prototype, prototype in enumerate(prototypes):
            for i_query, query in enumerate(X_q_embeddings):
                prototype, query = F.log_softmax(prototype, dim=-1), F.softmax(query, dim=-1)
                distance[i_prototype, i_query] = F.kl_div(prototype, query, reduction='batchmean')

    # print(f"distance.shape = {distance.shape}")

    labels_pred = train_lshot.lshot_prediction_labels(knn=knn, lmd=lambda_factor,
                                                      X=X_q_embeddings, unary=distance.transpose() ** 2,
                                                      support_label=X_s_labels.unique().numpy(),
                                                      logs=logs)

    results = np.full(shape=n_queries, fill_value=null_label)
    if remove_possibly_duplicates:
        # updates duplicates' labels
        new_labels_pred = np.zeros(shape=n_queries)
        np.put(new_labels_pred, non_duplicate_indices, labels_pred)
        for id1, id2 in similar_detections_indices:
            new_labels_pred[id2] = new_labels_pred[id1]
        results[X_q_non_null_indices] = new_labels_pred[X_q_non_null_indices]
    else:
        results[X_q_non_null_indices] = labels_pred


    return torch.from_numpy(results)

import itertools
from os.path import join

import numpy as np
from sklearn.metrics import f1_score

import torch

from tqdm import tqdm

from laplacianshot import train_lshot
from laplacianshot.plotting import plot_scatterplot, plot_distribution, plot_prototypes_difference
from laplacianshot.prototypes import get_prototypes_rectified, get_prototypes


def laplacian_shot(X_s_embeddings: torch.Tensor, X_s_labels: torch.Tensor,
                   X_q_embeddings: torch.Tensor, X_q_labels_pred: torch.Tensor, X_q_pred_confidence: torch.Tensor,
                   null_label: int = 20,
                   proto_rect: bool = True,
                   knn: int = None, lambda_factor: float = None,
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

    plot_scatterplot(embeddings_s=X_s_embeddings, labels_s=X_s_labels,
                     embeddings_q=X_q_embeddings,
                     title=f"before normalization", folder=join(".", "plots"))

    plot_distribution(distribution=X_q_pred_confidence,
                      title="Confidence of prediction", folder=join(".", "plots"))

    if logs:
        print(f"X_s_embeddings ranges in [{X_s_embeddings.float().min()}, {X_s_embeddings.float().max()}] "
              f"with mean norm {np.linalg.norm(X_s_embeddings.float(), 2, 1).mean()} before normalization")
        print(f"X_q_embeddings ranges in [{X_q_embeddings.float().min()}, {X_q_embeddings.float().max()}] "
              f"with mean norm {np.linalg.norm(X_q_embeddings.float(), 2, 1).mean()} before normalization")

    # normalizes the vectors
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

    plot_scatterplot(embeddings_s=X_s_embeddings, labels_s=X_s_labels,
                     embeddings_q=X_q_embeddings,
                     title=f"after normalization", folder=join(".", "plots"))

    if proto_rect:
        prototypes = get_prototypes_rectified(embeddings_s=X_s_embeddings, labels_s=X_s_labels,
                                              embeddings_q=X_q_embeddings, pseudolabels_q=X_q_labels_pred,
                                              confidence_q=X_q_pred_confidence)
    else:
        prototypes = get_prototypes(embeddings=X_s_embeddings, labels=X_s_labels)

    plot_scatterplot(embeddings_s=prototypes, labels_s=X_s_labels.unique(),
                     embeddings_q=X_q_embeddings,
                     title=f"of prototypes and queries", folder=join(".", "plots"))

    # tunes lambda
    if not lambda_factor or not knn:
        lambda_factor_best, knn_best = None, None
        scores = []
        lambda_factors = [lambda_factor] if lambda_factor \
            else np.linspace(start=0.1, stop=2, num=10, endpoint=True)
        knns = [knn] if knn \
            else np.linspace(start=1, stop=10, num=10, endpoint=True, dtype=int)
        combinations = list(itertools.product(lambda_factors, knns))
        for lambda_factor_try, knn_try in tqdm(combinations,
                                               desc=f"Tuning hyperparameters"):
            distance = np.linalg.norm(prototypes[:, None, :] - X_s_embeddings, 2, axis=-1)
            unary = distance.transpose() ** 2

            labels_pred_train = train_lshot.lshot_prediction_labels(knn=knn_try, lmd=lambda_factor_try,
                                                                    X=X_s_embeddings.numpy(), unary=unary,
                                                                    support_label=X_s_labels.unique().numpy(),
                                                                    logs=False)
            score = f1_score(y_true=X_s_labels, y_pred=labels_pred_train, average="macro")
            if not scores or score > np.max(scores):
                lambda_factor_best, knn_best = lambda_factor_try, knn_try
            scores += [score]
        if not lambda_factor:
            lambda_factor = lambda_factor_best
        if not knn:
            knn = knn_best
        if logs:
            print(
                f"Found parameters with best score {np.round(np.max(scores), 3)} (worst is {np.round(np.min(scores), 3)}):\n"
                f"knn = {knn_best}\tlambda_factor = {np.round(lambda_factor_best, 3)}")

    # gets predictions with null label
    X_q_null_indices = (X_q_labels_pred == null_label).nonzero().flatten()
    X_q_non_null_indices = (X_q_labels_pred != null_label).nonzero().flatten()

    # discards prediction with null label
    X_q_embeddings, X_q_labels_pred, X_q_pred_confidence = X_q_embeddings[X_q_non_null_indices], \
                                                           X_q_labels_pred[X_q_non_null_indices], \
                                                           X_q_pred_confidence[X_q_non_null_indices]

    # leverages fc classification results
    for i_embedding, (embedding, label, score) in enumerate(zip(X_q_embeddings,
                                                                X_q_labels_pred,
                                                                X_q_pred_confidence)):
        X_q_embeddings[i_embedding] = (1 - score) * embedding + \
                                      score * prototypes[label]

    plot_scatterplot(embeddings_s=prototypes, labels_s=X_s_labels.unique(),
                     embeddings_q=X_q_embeddings,
                     title=f"of prototypes and queries after leverage of induction",
                     folder=join(".", "plots"))

    plot_prototypes_difference(prototypes=get_prototypes(embeddings=X_s_embeddings, labels=X_s_labels),
                               prototypes_rectified=get_prototypes_rectified(
                                   embeddings_s=X_s_embeddings, labels_s=X_s_labels,
                                   embeddings_q=X_q_embeddings, pseudolabels_q=X_q_labels_pred,
                                   confidence_q=X_q_pred_confidence),
                               labels=X_s_labels.unique(),
                               folder=join(".", "plots"))
    # predicts the labels
    if logs:
        print(f"Predicting {len(X_q_embeddings)} labels with Laplacianshot")
    distance = np.linalg.norm(prototypes.numpy()[:, None, :] - X_q_embeddings.numpy(), 2, axis=-1)
    unary = distance.transpose() ** 2
    labels_pred = train_lshot.lshot_prediction_labels(knn=knn, lmd=lambda_factor,
                                                      X=X_q_embeddings, unary=unary,
                                                      support_label=X_s_labels.unique().numpy(),
                                                      logs=logs)

    # prepares labels for both null and non-null predictions
    results = np.zeros(n_queries)
    results[X_q_non_null_indices] = labels_pred
    results[X_q_null_indices] = null_label

    return torch.from_numpy(results)

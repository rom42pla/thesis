import numpy as np

import torch
import torch.nn.functional as F


def get_prototypes(embeddings: torch.Tensor, labels: torch.Tensor):
    prototypes = torch.zeros(size=(len(set(labels.tolist())), embeddings.shape[1]),
                             dtype=torch.float)
    for label in labels.unique():
        class_indices = (labels == label).nonzero().flatten()
        prototypes[label] = torch.mean(embeddings[class_indices], 0)
    return prototypes


def get_prototypes_rectified(embeddings_s: torch.Tensor, labels_s: torch.Tensor,
                             embeddings_q: torch.Tensor, pseudolabels_q: torch.Tensor,
                             confidence_q: torch.Tensor,
                             most_confident_qs: int = 10, epsilon: int = 10,
                             add_shifting_term: bool = False, normalize: bool = True):
    # retrieves base prototypes as mean of the supports
    base_prototypes = get_prototypes(embeddings=embeddings_s, labels=labels_s)

    # discards unconfident query predictions
    most_confident_embeddings_q, most_confident_pseudolabels_q = [], []
    for label in labels_s.unique():
        class_indices = (pseudolabels_q == label).nonzero().flatten()
        most_confident_indices = class_indices[torch.sort(confidence_q[class_indices],
                                                          descending=True)[1][:most_confident_qs]]
        most_confident_embeddings_q += [embeddings_q[most_confident_indices]]
        most_confident_pseudolabels_q += [pseudolabels_q[most_confident_indices]]
    embeddings_q, pseudolabels_q = torch.cat(most_confident_embeddings_q, dim=0), \
                                   torch.cat(most_confident_pseudolabels_q, dim=0)

    # shifts the queries towards the supports to reduce cross-class bias
    if add_shifting_term:
        shifting_term = (torch.sum(embeddings_s, dim=0) / len(embeddings_s)) - \
                        (torch.sum(embeddings_q, dim=0) / len(embeddings_q))
        embeddings_q += shifting_term

    # augments the data with pseudolabeled queries
    embeddings_augmented, labels_augmented = torch.cat((embeddings_s, embeddings_q), dim=0), \
                                             torch.cat((labels_s, pseudolabels_q), dim=0)

    # assigns the prototypes
    prototypes = torch.zeros_like(base_prototypes)
    for label in labels_augmented.unique():
        class_indices = (labels_augmented == label).nonzero().flatten()
        w = torch.exp(epsilon * F.cosine_similarity(embeddings_augmented[class_indices],
                                                    base_prototypes[label].expand(len(class_indices), -1),
                                                    dim=-1)) / \
            torch.sum(torch.exp(epsilon * F.cosine_similarity(embeddings_augmented[class_indices],
                                                              base_prototypes[label].expand(len(class_indices), -1),
                                                              dim=-1)))
        prototypes[label] = torch.sum(w.reshape(-1, 1) * embeddings_augmented[class_indices], dim=0)

    # normalizes the prototypes
    if normalize:
        prototypes = (prototypes - torch.mean(embeddings_s, dim=0)) / \
                     np.linalg.norm(prototypes, 2, 1)[:, None]

    return prototypes

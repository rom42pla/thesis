import copy
import itertools
import os
import time
import datetime
import logging
from os.path import join, exists
from typing import Dict, Union, Optional, List, Tuple
from collections import OrderedDict
import numpy as np

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torchvision import transforms as T

from detectron2.data import MetadataCatalog
from detectron2.data.common import AspectRatioGroupedDataset
from detectron2.structures import Instances, Boxes, ImageList
from detectron2.utils import comm
from torchvision.utils import save_image
from tqdm import tqdm

from fsdet.engine import DefaultTrainer
from fsdet.evaluation import (COCOEvaluator, DatasetEvaluators, LVISEvaluator, PascalVOCDetectionEvaluator,
                              DatasetEvaluator, print_csv_format, inference_context)

from pprint import pprint
import matplotlib.pyplot as plt

from laplacianshot import train_lshot


# def get_prototypes(data_loader: AspectRatioGroupedDataset, model: torch.nn.Module) \
#         -> Dict[int, Dict[str, Union[int, torch.Tensor]]]:
#     prototypes = {}
#     for batch in data_loader:
#         for img_data in batch:
#             img = img_data["image"]  # torch.Size([3, H, W])
#             boxes, labels = img_data["instances"].get_fields()["gt_boxes"].tensor.int(), \
#                             img_data["instances"].get_fields()["gt_classes"]
#             for box, label in zip(boxes, labels):
#                 # eventually adds the label to the dictionary
#                 if label.item() not in prototypes:
#                     prototypes[label.item()] = {
#                         "sum": None,
#                         "count": 0
#                     }
#                 # retrieves feature embeddings of each box
#                 img_crop = (img[:, box[1]:box[3], box[0]:box[2]] / 255).to(model.device)  # torch.Size([3, H, W])
#                 # todo figure out how do the backbone works
#                 features = model.backbone.bottom_up(img_crop.unsqueeze(0))["res5"][0, :, 0, 0] \
#                     .flatten()  # torch.Size([2048])
#                 # updates the prototypes
#                 if prototypes[label.item()]["sum"] is None:
#                     prototypes[label.item()]["sum"] = features
#                 else:
#                     prototypes[label.item()]["sum"] += features
#                 prototypes[label.item()]["count"] += 1
#                 # plt.imshow(img.permute(1, 2, 0))
#                 # plt.imshow(img_crop.permute(1, 2, 0))
#                 # plt.show()
#     prototypes = {k: v["sum"] / v["count"]
#                   for k, v in prototypes.items()}
#     return prototypes


def get_embeddings_and_labels(data_loader: AspectRatioGroupedDataset, model: torch.nn.Module) \
        -> torch.Tensor:
    embeddings, labels = [], []
    for batch in data_loader:
        for img_data in batch:
            img = img_data["image"]  # torch.Size([3, H, W])
            boxes, box_labels = img_data["instances"].get_fields()["gt_boxes"].tensor.int(), \
                                img_data["instances"].get_fields()["gt_classes"]
            for box, label in zip(boxes, box_labels):
                # retrieves feature embeddings of each box
                img_crop = (img[:, box[1]:box[3], box[0]:box[2]] / 255).to(model.device)  # torch.Size([3, H, W])
                # todo figure out how do the backbone works
                features = model.backbone.bottom_up(img_crop.unsqueeze(0))["res5"][0, :, 0, 0] \
                    .flatten()  # torch.Size([2048])
                # updates the embeddings
                embeddings += [features]
                labels += [label]
    # concatenates every embedding in a single tensor
    embeddings = torch.stack(embeddings)  # torch.Size([N, 2048])
    labels = torch.stack(labels)  # torch.Size([N])
    return embeddings, labels


class LaplacianTrainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(
                COCOEvaluator(dataset_name, cfg, True, output_folder)
            )
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test(cls, cfg, model, evaluators=None,
             data_augmentation: bool = False,
             use_laplacianshot: bool = True,
             embeddings_type: str = "embeddings",
             max_iters: Optional[int] = None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        assert isinstance(data_augmentation, bool)
        assert isinstance(use_laplacianshot, bool)
        assert isinstance(embeddings_type, str)
        assert embeddings_type in {"embeddings", "probabilities"}

        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(
                evaluators
            ), "{} != {}".format(len(cfg.DATASETS.TEST), len(evaluators))
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            dataloader_support = cls.build_train_loader(cfg, finite=True)
            dataloader_query = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model=model,
                                             dataloader_support=dataloader_support,
                                             dataloader_query=dataloader_query,
                                             evaluator=evaluator,
                                             data_augmentation=data_augmentation,
                                             use_laplacianshot=use_laplacianshot,
                                             embeddings_type=embeddings_type,
                                             max_iters=max_iters)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(
                        dataset_name
                    )
                )
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


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


def inference_on_dataset(model, dataloader_support, dataloader_query, evaluator,
                         use_laplacianshot: bool = True,
                         data_augmentation: bool = False,
                         embeddings_type: str = "embeddings",
                         max_iters: Optional[int] = None):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        dataloader_query: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    assert isinstance(data_augmentation, bool)
    assert isinstance(use_laplacianshot, bool)
    assert isinstance(embeddings_type, str)
    assert embeddings_type in {"embeddings", "probabilities"}
    embeddings_key = "box_features" \
        if embeddings_type == "embeddings" \
        else "pred_class_logits"
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    # logger.info("Start inference on {} images".format(len(data_loader)))
    print("Start inference on {} images".format(len(dataloader_query)))

    total = len(dataloader_query)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 100
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0

    inputs_total, outputs_total = [], []

    data_augment = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.05),
        T.RandomRotation(30)
    ])

    X_s_embeddings, X_s_labels = None, None
    X_q_embeddings = None
    with inference_context(model), torch.no_grad():
        # gets information from support set
        for img_data in tqdm(dataloader_support.dataset.dataset, desc=f"Getting support features"):
            # retrieves the full image
            img = img_data["image"].to(model.device)  # torch.Size([3, H, W])
            # casts the image from [0, 255] to [0, 1]
            # img = (img / 255).float()
            # corrects colors from BGR to RGB
            # img = img[[2, 1, 0], :, :]

            # images_folder = join("debug", "images")
            # if not exists(images_folder):
            #     os.makedirs(images_folder)
            # save_image(img, join(images_folder, f"{img_data['image_id']}.png"))

            # retrieves data about found boxes
            boxes_labels = img_data["instances"].get_fields()["gt_classes"].to(model.device)  # torch.Size([1])
            boxes_coords = img_data["instances"].get_fields()["gt_boxes"].tensor.to(model.device)  # torch.Size([1, 4])
            results = []
            # loops over found boxes
            for box_coords in boxes_coords:
                # crops the image according to box's detection
                img_boxed = img[:,
                            int(box_coords[1]): int(box_coords[3]),
                            int(box_coords[0]): int(box_coords[2])]
                # creates the box proposal query for the classification
                proposal = [
                    Instances(image_size=img_boxed.shape[-2:],
                              objectness_logits=[torch.tensor([1], device=model.device)],
                              proposal_boxes=Boxes(box_coords.unsqueeze(0)))
                ]
                # retrieves embeddings and prediction scores
                if data_augmentation:
                    for i in range(5):
                        img_augmented = data_augment(img_boxed)
                        img_augmented = normalize_image(img=img_augmented, model=model)
                        # features = model.backbone(img_preprocessing(img_augmented.unsqueeze(0)))
                        features = model.backbone(img_augmented.unsqueeze(0))
                        results += [model.roi_heads(img_augmented, features, proposal, None)[0]]
                else:
                    # images_folder = join("debug", "images")
                    # if not exists(images_folder):
                    #     os.makedirs(images_folder)
                    # save_image(img_boxed, join(images_folder, f"{img_data['image_id']}_boxed.png"))
                    # features = model.backbone(img_preprocessing(img_boxed.unsqueeze(0)))
                    img_boxed = normalize_image(img=img_boxed, model=model)
                    features = model.backbone(img_boxed.unsqueeze(0))
                    results += [model.roi_heads(img_data, features, proposal, None)[0]]
            for result in results:
                # updates X_s_embeddings
                features = result[0].get(embeddings_key)[0].unsqueeze(0)
                if embeddings_type == "probabilities":
                    features = F.softmax(features, dim=-1)
                features = features.type(torch.half).cpu()
                X_s_embeddings = torch.cat([X_s_embeddings, features], dim=0) \
                    if isinstance(X_s_embeddings, torch.Tensor) \
                    else features
                # updates X_s_labels
                labels = boxes_labels.type(torch.short).flatten().cpu()
                X_s_labels = torch.cat([X_s_labels, labels]) \
                    if isinstance(X_s_labels, torch.Tensor) \
                    else labels

        # gets informations from query set
        for idx, inputs in enumerate(dataloader_query):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            if max_iters and idx >= max_iters:
                break

            start_compute_time = time.time()
            outputs = model(inputs)

            # updates X_q_embeddings
            features = torch.stack([b
                                    for s in outputs
                                    for b in s["instances"].get_fields()[embeddings_key]])
            if embeddings_type == "probabilities":
                features = F.softmax(features, dim=0)
            features = features.type(torch.half).cpu()
            outputs[0]["instances"].remove("box_features")
            outputs[0]["instances"].remove("pred_class_logits")
            X_q_embeddings = torch.cat([X_q_embeddings, features], dim=0) \
                if isinstance(X_q_embeddings, torch.Tensor) \
                else features

            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            # outputs[0]['instances'].get_fields()["pred_classes"][0] = 1
            # evaluator.process(inputs, outputs)

            # cleanses inputs and outputs before collection
            for i, (input, output) in enumerate(zip(inputs, outputs)):
                for k, v in input.items():
                    if isinstance(v, torch.Tensor):
                        inputs[i][k] = v.to("cpu")
                # img = input["image"]
                for k, v in output.items():
                    #     if isinstance(v, Instances):
                    #         boxes = output["instances"].get_fields()["pred_boxes"].tensor.int()
                    #         box_features = []
                    #         for i_box, box in enumerate(boxes):
                    #             # retrieves feature embeddings of each box
                    #             img_crop = (img[:, box[1]:box[3], box[0]:box[2]] / 255).to(
                    #                 model.device)  # torch.Size([3, H, W])
                    #             if 0 in img_crop.shape:
                    #                 img_crop = torch.rand(size=(3, 224, 224), device=model.device)
                    #             box_features += [resnet18(img_preprocessing(img_crop.unsqueeze(0))).flatten()]
                    #         outputs[i][k].set("box_features", torch.stack(box_features))
                    #
                    if isinstance(v, torch.Tensor) or isinstance(v, Instances):
                        outputs[i][k] = v.to("cpu")

            # collects predictions
            inputs_total += inputs
            outputs_total += outputs

            # print(len(outputs), len(outputs_total), idx)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                print(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))

    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )

    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    print(
        f"X_s_embeddings ranges in [{X_s_embeddings.float().min()}, {X_s_embeddings.float().max()}] with mean norm {np.linalg.norm(X_s_embeddings, 2, 1).mean()}")
    print(
        f"X_q_embeddings ranges in [{X_q_embeddings.float().min()}, {X_q_embeddings.float().max()}] with mean norm {np.linalg.norm(X_q_embeddings, 2, 1).mean()}")
    # exit()

    # evaluates the results
    if use_laplacianshot:
        print(f"Predicting labels with LaplacianShot using {embeddings_type}")
        X_q_labels_laplacian = laplacian_shot(X_s_embeddings=X_s_embeddings,
                                              X_s_labels=X_s_labels,
                                              X_q_embeddings=X_q_embeddings,
                                              knn=None, lambda_factor=None)
        for i, (input, output) in enumerate(zip(inputs_total, outputs_total)):
            # replaces fully connected layer's labels with laplacian's ones
            instances = len(output["instances"].get_fields()["pred_classes"])
            output["instances"].set(name="pred_classes",
                                    value=X_q_labels_laplacian[i * instances:i * instances + instances])
            # evaluates the results
            evaluator.process([input], [output])
            # print(output["instances"].get_fields()["pred_classes"])
            # print(i, instances)
            # print(X_q_labels_laplacian[i * instances:i * instances + instances])
            # exit()
            # processes the outputs
    else:
        print(f"Predicting labels using classification layer")
        for i, (input, output) in enumerate(zip(inputs_total, outputs_total)):
            evaluator.process([input], [output])

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def laplacian_shot(X_s_embeddings: torch.Tensor, X_s_labels: torch.Tensor,
                   X_q_embeddings: torch.Tensor,
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

    def get_prototypes(embeddings: torch.Tensor, labels: torch.Tensor):
        prototypes = np.zeros(shape=(len(set(labels.tolist())), embeddings.shape[1]),
                              dtype=float)
        counter = np.zeros(shape=(len(set(labels.tolist())),),
                           dtype=int)
        for embedding, label in zip(embeddings, labels):
            prototypes[label] += embedding
            counter[label] += 1
        prototypes = prototypes / counter[:, None]
        return prototypes

    def get_metric(metric_type):
        METRICS = {
            'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
            'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
            'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
            'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
        }
        return METRICS[metric_type]

    # # builds the dictionary of labels
    # labels_dict = {label: index for index, label in enumerate(set(X_s_labels.tolist()))}
    # # converts X_s_labels according to the dictionary
    # X_s_labels_encoded = copy.deepcopy(X_s_labels)
    # for i_label, label in enumerate(X_s_labels):
    #     X_s_labels_encoded[i_label] = labels_dict[label.item()]
    # # retrieves prototypes
    # print(f"Computing prototypes")
    # m = get_prototypes(embeddings=X_s_embeddings, labels=X_s_labels_encoded)
    # print(f"Computing distances from the {X_q_embeddings.shape[0]} queries to the {m.shape[0]} prototypes")
    # a = get_distances(embeddings=X_q_embeddings, prototypes=m)
    # print(f"Initializing Y matrix")
    # Y = torch.zeros(size=(X_q_embeddings.shape[0], m.shape[0]),
    #                 dtype=torch.float32, device=X_q_embeddings.device)
    # for i_query, _ in enumerate(Y):
    #     Y[i_query] = torch.exp(-a[i_query]) / \
    #                  torch.dot(torch.ones_like(a[i_query]).T, torch.exp(-a[i_query]))
    # print(f"Entering main loop")
    # for _ in range(2):
    #     for i_query, _ in enumerate(Y):
    #         Y[i_query] = torch.exp(-a[i_query]) / \
    #                      torch.dot(torch.ones_like(a[i_query]).T, torch.exp(-a[i_query]))
    #
    # print(Y)
    # y = torch.exp(-a) / torch.dot(torch.ones_like(a).T, torch.exp(-a))

    # gets query's distances from prototypes
    labels_uniques = X_s_labels[np.sort(np.unique(X_s_labels, return_index=True)[1])]
    # prototypes = get_prototypes(embeddings=X_s_embeddings, labels=X_s_labels)
    X_s_embeddings = X_s_embeddings.numpy().astype(float)
    X_q_embeddings = X_q_embeddings.numpy().astype(float)

    mean = X_s_embeddings.mean()
    # X_s_embeddings = X_s_embeddings - mean
    X_s_embeddings = X_s_embeddings - X_s_embeddings.mean()
    X_s_embeddings = X_s_embeddings / np.linalg.norm(X_s_embeddings, 2, 1)[:, None]
    # X_q_embeddings = X_q_embeddings - mean
    X_q_embeddings = X_q_embeddings - X_q_embeddings.mean()
    X_q_embeddings = X_q_embeddings / np.linalg.norm(X_q_embeddings, 2, 1)[:, None]

    # def plot_scatterplot(X, labels):
    #     import seaborn as sns
    #     import pandas as pd
    #     pca = PCA(n_components=2)
    #     pca.fit(X)
    #
    #     df = pd.DataFrame([{
    #         "x": x,
    #         "y": y,
    #         "label": label.item()
    #     } for (x, y), label in zip(pca.transform(X), labels)])
    #
    #     df = df.groupby("label").mean()
    #
    #     fig, ax = plt.subplots(1, figsize=[10, 10])
    #     sns.scatterplot(x="x", y="y", hue="label", data=df,
    #                     palette="Paired", ax=ax).set_title(f"Scatterplot of X_s")
    #     ax.set_xlabel("")
    #     ax.set_ylabel("")
    #     plt.savefig("scatterplot.png")

    # plot_scatterplot(X=X_s_embeddings, labels=X_s_labels)

    # proto_rect = True

    if proto_rect:
        # preprocesses the inputs
        prototypes = X_s_embeddings
        # X_q_embeddings = X_q_embeddings / np.linalg.norm(X_q_embeddings, 2, 1)[:, None]

        eta = prototypes.mean(0) - X_q_embeddings.mean(0)  # shift
        X_q_embeddings = X_q_embeddings + eta[np.newaxis, :]

        query_aug = np.concatenate((prototypes, X_q_embeddings), axis=0)
        gallery_ = prototypes.reshape(len(labels_uniques), 2, prototypes.shape[-1]).mean(1)
        gallery_ = torch.from_numpy(gallery_)
        query_aug = torch.from_numpy(query_aug)
        distance = get_metric('cosine')(gallery_, query_aug)
        predict = torch.argmin(distance, dim=1)
        cos_sim = F.cosine_similarity(query_aug[:, None, :], gallery_[None, :, :], dim=2)
        cos_sim = 10 * cos_sim
        W = F.softmax(cos_sim, dim=1)
        gallery_list = [(W[predict == i, i].unsqueeze(1) * query_aug[predict == i]).mean(0, keepdim=True)
                        for i in predict.unique()]
        prototypes = torch.cat(gallery_list, dim=0).numpy()
    else:
        prototypes = get_prototypes(embeddings=X_s_embeddings, labels=X_s_labels)

    # tunes lambda
    if not lambda_factor or not knn:
        lambda_factor_best, knn_best = None, None
        scores = []
        lambda_factors, knns = np.linspace(start=0.1, stop=2, num=10, endpoint=True), \
                               np.linspace(start=1, stop=10, num=10, endpoint=True, dtype=int)
        combinations = list(itertools.product(lambda_factors, knns))
        for lambda_factor_try, knn_try in tqdm(combinations,
                                               desc=f"Tuning hyperparameters"):
            subtract = prototypes[:, None, :] - X_s_embeddings
            distance = np.linalg.norm(subtract, 2, axis=-1)
            unary = distance.transpose() ** 2

            labels_pred_train = train_lshot.lshot_prediction_labels(knn=knn_try, lmd=lambda_factor_try,
                                                                    X=X_s_embeddings, unary=unary,
                                                                    support_label=labels_uniques,
                                                                    logs=False)
            from sklearn.metrics import f1_score
            score = f1_score(y_true=X_s_labels.numpy(), y_pred=labels_pred_train, average="macro")
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

    # predicts the labels
    subtract = prototypes[:, None, :] - X_q_embeddings
    distance = np.linalg.norm(subtract, 2, axis=-1)
    unary = distance.transpose() ** 2

    if logs:
        print(f"Predicting {len(X_q_embeddings)} labels with Laplacianshot")
    labels_pred = train_lshot.lshot_prediction_labels(knn=knn, lmd=lambda_factor,
                                                      X=X_q_embeddings, unary=unary,
                                                      support_label=labels_uniques,
                                                      logs=logs)
    return labels_pred

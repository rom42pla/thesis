import copy
import os
import time
import datetime
import logging
from os.path import join
from typing import Dict, Union, Optional, List, Tuple
from collections import OrderedDict
import numpy as np

import torch
from torchvision import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.data.common import AspectRatioGroupedDataset
from detectron2.structures import Instances, Boxes
from detectron2.utils import comm
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
    def test(cls, cfg, model,
             evaluators=None, max_iters: Optional[int] = None):
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
                                             evaluator=evaluator, max_iters=max_iters)
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


def inference_on_dataset(model, dataloader_support, dataloader_query, evaluator,
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

    features_support, labels_support = [], []
    inputs_total, outputs_total = [], []

    # todo replace with rcnn's backbone
    import torchvision.models as models
    from torch import nn
    resnet18 = nn.Sequential(
        *list(models.resnet18(pretrained=True).children())[:-1]
    ).to(model.device)
    resnet18.eval()
    for p in resnet18.parameters():
        p.requires_grad = False

    img_preprocessing = T.Compose([
        T.Resize((224, 224)),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    with inference_context(model), torch.no_grad():
        # gets information from support set
        for img_data in tqdm(dataloader_support.dataset.dataset, desc=f"Getting support features"):
            img = img_data["image"].to(model.device) / 255  # torch.Size([3, H, W])
            boxes, box_labels = img_data["instances"].get_fields()["gt_boxes"].tensor.int(), \
                                img_data["instances"].get_fields()["gt_classes"]
            features = model.backbone(img_preprocessing(img.unsqueeze(0)))
            single_proposal = [
                Instances(image_size=img.shape[-2:],
                          objectness_logits=[torch.tensor([0], device=model.device)],
                          proposal_boxes=Boxes(
                              torch.tensor([[0, 0, img.shape[-1] - 1, img.shape[-2] - 1]],
                                           device=model.device)))
            ]
            results, _ = model.roi_heads(img_data, features, single_proposal, None)
            features_support += [results[0].get("box_features")[0].cpu()]
            labels_support += [box_labels.cpu()]
        #     print(torch.stack(features_support).shape)
        #     print(box_labels)
        #     exit()
        #     for box, label in zip(boxes, box_labels):
        #         # retrieves feature embeddings of each box
        #         img_crop = (img[:, box[1]:box[3], box[0]:box[2]] / 255).to(model.device)  # torch.Size([3, H, W])
        #         img_crop = img_preprocessing(img_crop)  # torch.Size([3, 224, 224])
        #         support_imgs_crops += [img_crop]
        #         labels_support += [label]
        #         # todo figure out how do the backbone works
        #         # features = model.backbone.bottom_up(img_crop.unsqueeze(0))["res5"][0, :, 0, 0].flatten()  # torch.Size([2048])
        # features = resnet18(torch.stack(support_imgs_crops))
        # features = features.view((features.shape[0], -1)).to("cpu")  # torch.Size([B, 512])
        # features_support += [f for f in features]

        # gets informations from query set
        for idx, inputs in enumerate(dataloader_query):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            if max_iters and idx >= max_iters:
                break

            start_compute_time = time.time()
            outputs = model(inputs)

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

    logger.info(
        f"Predicting labels using LaplacianShot"
    )
    X_q_labels_laplacian = laplacian_shot(X_s_embeddings=torch.stack(features_support),
                                          X_s_labels=torch.stack(labels_support).flatten(),
                                          X_q_embeddings=torch.stack([b
                                                                      for s in outputs_total
                                                                      for b in
                                                                      s["instances"].get_fields()["box_features"]]),
                                          knn=3, lambda_factor=0.1)
    # print(X_q_labels_laplacian.shape)
    # print(outputs_total[0])
    # print(outputs_total[0]["instances"].get_fields()["pred_classes"])
    # outputs_total[0]["instances"].set(name="pred_classes", value=X_q_labels_laplacian[:100])
    # print(outputs_total[0]["instances"].get_fields()["pred_classes"])
    # exit()

    # evaluates the results
    for i, (input, output) in enumerate(zip(inputs_total, outputs_total)):
        # replaces fully connected layer's labels with laplacian's ones
        # instances = len(output["instances"].get_fields()["pred_classes"])
        # output["instances"].set(name="pred_classes", value=X_q_labels_laplacian[i*instances:i*instances + instances])
        # processes the outputs
        evaluator.process([input], [output])
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


# def get_distances(embeddings: torch.Tensor, prototypes: torch.Tensor):
#     assert embeddings.shape[1] == prototypes.shape[1], f"Inconsistent embeddings' shape " \
#                                                        f"between embeddings {embeddings.shape[1]} " \
#                                                        f"and prototypes {prototypes.shape[1]}"
#     distances = torch.zeros(size=(embeddings.shape[0], prototypes.shape[0]),
#                             dtype=torch.float32, device=embeddings.device)
#     for i_embedding, embedding in enumerate(embeddings):
#         for i_prototype, prototype in enumerate(prototypes):
#             distances[i_embedding][i_prototype] = torch.dist(embedding, prototype, p=2)
#     return distances


def laplacian_shot(X_s_embeddings: torch.Tensor, X_s_labels: torch.Tensor,
                   X_q_embeddings: torch.Tensor,
                   knn: int = 3, lambda_factor: float = 0.1) -> torch.Tensor:
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
    assert isinstance(knn, int) and knn >= 1, f"knn parameter must be an integer >= 1"
    assert lambda_factor >= 0, f"lambda_factor must be >= 0 " \
                               f"but got {lambda_factor}"

    def get_prototypes(embeddings: torch.Tensor, labels: torch.Tensor):
        prototypes = torch.zeros(size=(len(set(labels.tolist())), embeddings.shape[1]),
                                 dtype=torch.float32, device=embeddings.device)
        counter = torch.zeros(size=(len(set(labels.tolist())),),
                              dtype=torch.int, device=embeddings.device)
        for embedding, label in zip(embeddings, labels):
            prototypes[label] += embedding
            counter[label] += 1
        prototypes = prototypes / counter[:, None]
        return prototypes

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
    subtract = get_prototypes(embeddings=X_s_embeddings, labels=X_s_labels)[:, None, :] - X_q_embeddings
    distance = np.linalg.norm(subtract, 2, axis=-1)
    unary = distance.transpose() ** 2

    # predicts the labels
    labels_pred = train_lshot.lshot_prediction_labels(knn=3, lmd=lambda_factor,
                                                      X=X_q_embeddings, unary=unary,
                                                      support_label=X_s_labels[np.sort(
                                                          np.unique(X_s_labels, return_index=True)[1])])
    return labels_pred

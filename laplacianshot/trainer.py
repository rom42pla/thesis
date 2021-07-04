import os
import time
import datetime
import logging
from typing import Optional
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T

from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes, ImageList
from detectron2.utils import comm
from tqdm import tqdm

from fsdet.engine import DefaultTrainer
from fsdet.evaluation import (COCOEvaluator, DatasetEvaluators, LVISEvaluator, PascalVOCDetectionEvaluator,
                              DatasetEvaluator, print_csv_format, inference_context)
from laplacianshot.inference import laplacian_shot


class LaplacianTrainer(DefaultTrainer):
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
             proto_rect: bool = True,
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
        assert isinstance(proto_rect, bool)
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
                                             proto_rect=proto_rect,
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
                         proto_rect: bool = True,
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
    X_q_embeddings, X_q_labels_pred, X_q_labels_pred_scores, X_q_pred_confidence = None, None, None, None
    with inference_context(model), torch.no_grad():

        # =======#=======#=======#=======#=======#=======#=======#=======#=======
        # ======= S U P P O R T
        # =======#=======#=======#=======#=======#=======#=======#=======#=======
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
                # img_boxed = img[:,
                #             int(box_coords[1]): int(box_coords[3]),
                #             int(box_coords[0]): int(box_coords[2])]
                # creates the box proposal query for the classification
                proposal = [
                    Instances(image_size=img.shape[-2:],
                              # objectness_logits=[torch.tensor([1], device=model.device)],
                              proposal_boxes=Boxes(box_coords.unsqueeze(0)))
                ]
                # retrieves embeddings and prediction scores
                if data_augmentation:
                    for i in range(5):
                        img_augmented = normalize_image(img=data_augment(img), model=model)
                        # features = model.backbone(img_preprocessing(img_augmented.unsqueeze(0)))
                        features = model.backbone(img_augmented.unsqueeze(0))
                        results += [model.roi_heads(img_augmented, features, proposal)[0]]
                else:
                    # images_folder = join("debug", "images")
                    # if not exists(images_folder):
                    #     os.makedirs(images_folder)
                    # save_image(img_boxed, join(images_folder, f"{img_data['image_id']}_boxed.png"))
                    # features = model.backbone(img_preprocessing(img_boxed.unsqueeze(0)))
                    img = normalize_image(img=img, model=model)
                    features = model.backbone(img.unsqueeze(0))
                    results += [model.roi_heads(img_data, features, proposal)[0]]

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

        # =======#=======#=======#=======#=======#=======#=======#=======#=======
        # ======= Q U E R Y
        # =======#=======#=======#=======#=======#=======#=======#=======#=======
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
                                    for b in s["instances"].get_fields()[embeddings_key]]).cpu()
            labels_pred = torch.stack([b
                                       for s in outputs
                                       for b in s["instances"].get_fields()["pred_classes"]]).cpu()
            labels_pred_scores = torch.stack([b
                                              for s in outputs
                                              for b in s["instances"].get_fields()["pred_class_logits"]]).cpu()
            pred_confidence = torch.stack([b
                                           for s in outputs
                                           for b in s["instances"].get_fields()["scores"]]).cpu()
            labels_pred_scores = F.softmax(labels_pred_scores, dim=0)
            if embeddings_type == "probabilities":
                features = F.softmax(features, dim=0)
            outputs[0]["instances"].remove("box_features")
            outputs[0]["instances"].remove("pred_class_logits")

            X_q_embeddings = torch.cat([X_q_embeddings, features], dim=0) \
                if isinstance(X_q_embeddings, torch.Tensor) \
                else features
            X_q_labels_pred = torch.cat([X_q_labels_pred, labels_pred], dim=0) \
                if isinstance(X_q_labels_pred, torch.Tensor) \
                else labels_pred
            X_q_labels_pred_scores = torch.cat([X_q_labels_pred_scores, labels_pred_scores], dim=0) \
                if isinstance(X_q_labels_pred_scores, torch.Tensor) \
                else labels_pred_scores
            X_q_pred_confidence = torch.cat([X_q_pred_confidence, pred_confidence], dim=0) \
                if isinstance(X_q_pred_confidence, torch.Tensor) \
                else pred_confidence

            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time

            # cleanses inputs and outputs before collection
            for i, (input, output) in enumerate(zip(inputs, outputs)):
                for k, v in input.items():
                    if isinstance(v, torch.Tensor):
                        inputs[i][k] = v.to("cpu")
                # img = input["image"]
                for k, v in output.items():
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
    # exit()

    # evaluates the results
    if use_laplacianshot:
        print(f"Predicting labels with LaplacianShot using {embeddings_type}")
        X_q_labels_laplacian = laplacian_shot(X_s_embeddings=X_s_embeddings,
                                              X_s_labels=X_s_labels,
                                              X_q_embeddings=X_q_embeddings,
                                              X_q_labels_pred=X_q_labels_pred,
                                              X_q_pred_confidence=X_q_pred_confidence,
                                              proto_rect=proto_rect,
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
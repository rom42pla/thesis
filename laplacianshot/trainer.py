import itertools
import os
import time
from os.path import join, splitext, basename, exists, isfile
import logging
from copy import deepcopy
from typing import Optional
from collections import OrderedDict

import pandas as pd
from compress_pickle import dump, load

import torch
from torch import nn

from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes
from detectron2.utils import comm
from tqdm import tqdm

from fsdet.engine import DefaultTrainer
from fsdet.evaluation import (COCOEvaluator, DatasetEvaluators, LVISEvaluator, PascalVOCDetectionEvaluator,
                              DatasetEvaluator, print_csv_format, inference_context)
from laplacianshot.images_manipulation import normalize_image
from laplacianshot.inference import laplacian_shot
from laplacianshot.plotting import plot_detections, plot_supports


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
             use_laplacianshot: bool = True,
             use_classification_layer: bool = True,
             rectify_prototypes: Optional[bool] = True,
             leverage_classification: Optional[bool] = True,
             embeddings_type: Optional[str] = "embeddings",
             max_iters: Optional[int] = None,
             laplacianshot_logs: bool = True,
             save_checkpoints: bool = True):
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
        assert isinstance(use_laplacianshot, bool)
        assert embeddings_type in {None, "embeddings", "probabilities"}

        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(
                evaluators
            ), "{} != {}".format(len(cfg.DATASETS.TEST), len(evaluators))
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            dataloader_support = cls.build_train_loader(cfg)
            dataloader_query = cls.build_test_loader(cfg, dataset_name)

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
                                             use_laplacianshot=use_laplacianshot,
                                             use_classification_layer=use_classification_layer,
                                             rectify_prototypes=rectify_prototypes,
                                             leverage_classification=leverage_classification,
                                             embeddings_type=embeddings_type,
                                             max_iters=max_iters,
                                             cfg=cfg,
                                             save_checkpoints=save_checkpoints,
                                             laplacianshot_logs=laplacianshot_logs)
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
                         use_laplacianshot: bool = True,
                         use_classification_layer: bool = True,
                         rectify_prototypes: bool = True,
                         leverage_classification: bool = True,
                         embeddings_type: Optional[str] = "embeddings",
                         max_iters: Optional[int] = None,
                         cfg=None,
                         save_checkpoints: bool = True,
                         laplacianshot_logs: bool = True):
    assert isinstance(use_laplacianshot, bool)
    assert embeddings_type in {None, "embeddings", "probabilities"}

    assert use_laplacianshot or use_classification_layer
    if use_laplacianshot:
        assert isinstance(laplacianshot_logs, bool)

    n_query_images = max_iters if max_iters \
        else len(dataloader_query)
    evaluator.reset()

    inputs_agg, outputs_agg = [], []
    X_s_embeddings, X_s_probabilities, X_s_labels, X_s_imgs = [], [], [], []

    test_score_thresh_original, test_detections_per_img_original = model.roi_heads.test_score_thresh, \
                                                                   model.roi_heads.test_detections_per_img

    datasets_names = "_".join(cfg.DATASETS.TRAIN)
    model_name = basename(
        splitext(cfg.MODEL.WEIGHTS)[0]
    )
    checkpoint_filename = join("checkpoints", "_".join([model_name, datasets_names]) + ".bz")

    times = pd.DataFrame()
    with inference_context(model), torch.no_grad():

        # =======#=======#=======#=======#=======#=======#=======#=======#=======
        # ======= S U P P O R T
        # =======#=======#=======#=======#=======#=======#=======#=======#=======
        if use_laplacianshot:
            starting_time = time.time()
            # sets the model for support predictions
            model.roi_heads.test_score_thresh, model.roi_heads.test_detections_per_img = 0, 1

            for img_data in tqdm(dataloader_support.dataset.dataset, desc=f"Getting support data"):
                # retrieves the full image
                img = img_data["image"].to(model.device)  # torch.Size([3, H, W])
                img_normalized = normalize_image(img=img, model=model)
                # features = model.backbone(img.float().unsqueeze(0))
                features = model.backbone(img_normalized.unsqueeze(0))

                # retrieves data about found boxes
                boxes_labels = img_data["instances"].get_fields()["gt_classes"].to(model.device)  # torch.Size([1])
                boxes_coords = img_data["instances"].get_fields()["gt_boxes"].tensor.to(
                    model.device)  # torch.Size([1, 4])

                # loops over found boxes
                for box, label in zip(boxes_coords, boxes_labels):
                    # creates the box proposal query for the classification
                    proposal = [
                        Instances(image_size=img.shape[-2:],
                                  # objectness_logits=[torch.tensor([1], device=model.device)],
                                  proposal_boxes=Boxes(box.unsqueeze(0)))
                    ]
                    # retrieves embeddings and prediction scores
                    result = model.roi_heads(img_data, features, proposal)[0][0]
                    if len(result.get("box_features")) == 0:
                        continue
                    features = result.get("box_features")[0].type(torch.half)
                    scores = result.get("pred_class_logits")[0].type(torch.half)

                    # keeps relevant infos
                    X_s_imgs += [img[:,
                                 int(box[1]): int(box[3]),
                                 int(box[0]): int(box[2])].cpu()]
                    X_s_embeddings += [features.cpu()]
                    X_s_probabilities += [scores.cpu()]
                    X_s_labels += [label.cpu()]

            X_s_embeddings, X_s_probabilities, X_s_labels = torch.stack(X_s_embeddings, dim=0), \
                                                            torch.stack(X_s_probabilities, dim=0), \
                                                            torch.stack(X_s_labels, dim=0)

            plot_supports(imgs=X_s_imgs, labels=X_s_labels, folder="plots")

            # resets the model
            model.roi_heads.test_score_thresh, model.roi_heads.test_detections_per_img = test_score_thresh_original, \
                                                                                         test_detections_per_img_original
            # records the times
            times = times.append(
                {
                    "phase": "support features retrieval",
                    "time_from": int(starting_time),
                    "time_to": int(time.time()),
                    "time_elapsed": int(time.time() - starting_time)
                },
                ignore_index=True)

        # =======#=======#=======#=======#=======#=======#=======#=======#=======
        # ======= Q U E R Y
        # =======#=======#=======#=======#=======#=======#=======#=======#=======
        # eventually loads the checkpoints from memory
        if exists(checkpoint_filename) and not max_iters:
            starting_time = time.time()

            evaluator._logger.info(f"Loading inputs and outputs from {checkpoint_filename}")
            inputs_agg, outputs_agg = load(checkpoint_filename)

            # records the times
            times = times.append(
                {
                    "phase": "query checkpoint loading",
                    "time_from": int(starting_time),
                    "time_to": int(time.time()),
                    "time_elapsed": int(time.time() - starting_time)
                },
                ignore_index=True)
        else:
            starting_time = time.time()
            for i_query, inputs in tqdm(enumerate(dataloader_query), desc=f"Predicting query data",
                                        total=n_query_images):
                # eventually early stops the computation
                if max_iters and i_query >= max_iters:
                    break

                outputs = model(inputs)
                torch.cuda.synchronize()

                # cleanses inputs and outputs before collection
                for i_output, (input, output) in enumerate(zip(inputs, outputs)):
                    for k, v in input.items():
                        if isinstance(v, torch.Tensor):
                            inputs[i_output][k] = v.to("cpu")
                    for k, v in output.items():
                        if isinstance(v, torch.Tensor) or isinstance(v, Instances):
                            outputs[i_output][k] = v.to("cpu")

                # plots a sample of detection
                if i_query == 0:
                    plot_detections(img=inputs[0]["image"],
                                    boxes=outputs[0]["instances"].get_fields()["pred_boxes"].tensor,
                                    confidences=outputs[0]["instances"].get_fields()["scores"],
                                    labels=outputs[0]["instances"].get_fields()["pred_classes"],
                                    folder="plots")
                # slims inputs
                inputs = [
                    {
                        k: v
                        for k, v in input.items()
                        if k != "image"
                    }
                    for input in inputs
                ]

                # collects predictions
                inputs_agg += inputs
                outputs_agg += outputs

            # records the times
            times = times.append(
                {
                    "phase": "query features retrieval",
                    "time_from": int(starting_time),
                    "time_to": int(time.time()),
                    "time_elapsed": int(time.time() - starting_time)
                },
                ignore_index=True)

            if save_checkpoints and not max_iters:
                starting_time = time.time()

                evaluator._logger.info(f"Compressing checkpoint in {checkpoint_filename}")
                dump((inputs_agg, outputs_agg), checkpoint_filename)

                # records the times
                times = times.append(
                    {
                        "phase": "query checkpoint saving",
                        "time_from": int(starting_time),
                        "time_to": int(time.time()),
                        "time_elapsed": int(time.time() - starting_time)
                    },
                    ignore_index=True)

    final_results = pd.DataFrame()

    # evaluates the results using the classification layer
    if use_classification_layer:
        starting_time = time.time()

        evaluator._logger.info(f"Predicting labels using classification layer")
        evaluator.reset()
        for i_query, (input, output) in enumerate(zip(inputs_agg, outputs_agg)):
            evaluator.process([input], [output])
        evaluation_results = evaluator.evaluate()
        final_results = final_results.append(
            {
                "use_classification_layer": use_classification_layer,
                "use_laplacianshot": use_laplacianshot,
                **dict(evaluation_results)["bbox"]
            },
            ignore_index=True
        )

        # records the times
        times = times.append(
            {
                "phase": "model classification",
                "time_from": int(starting_time),
                "time_to": int(time.time()),
                "time_elapsed": int(time.time() - starting_time)
            },
            ignore_index=True)

    # evaluates the results using laplacianshot
    if use_laplacianshot:
        starting_time = time.time()

        combinations = itertools.product(
            [True, False] if not rectify_prototypes else [rectify_prototypes],
            [True, False] if not leverage_classification else [leverage_classification],
            ["embeddings", "probabilities"] if not embeddings_type else [embeddings_type]
        )
        for rectify_prototypes, leverage_classification, embeddings_type in combinations:
            evaluator._logger.info(f"Predicting labels with LaplacianShot using {embeddings_type}")
            embeddings_key = "box_features" if embeddings_type == "embeddings" else "pred_class_logits"
            X_q_labels_laplacian = laplacian_shot(
                X_s_embeddings=(
                    X_s_embeddings if embeddings_type == "embeddings" else X_s_probabilities).detach().clone(),
                X_s_labels=X_s_labels.detach().clone(),
                X_q_embeddings=torch.stack([field
                                            for instance_output in outputs_agg
                                            for field in instance_output["instances"].get_fields()[embeddings_key]]),
                X_q_labels_pred=torch.stack([field
                                             for instance_output in outputs_agg
                                             for field in instance_output["instances"].get_fields()["pred_classes"]]),
                X_q_pred_confidence=torch.stack([field
                                                 for instance_output in outputs_agg
                                                 for field in instance_output["instances"].get_fields()["scores"]]),
                proto_rect=rectify_prototypes,
                leverage_classification=leverage_classification,
                embeddings_are_probabilities=True if embeddings_type == "probabilities" else False,
                knn=None, lambda_factor=None, logs=laplacianshot_logs)
            # evaluates the results
            evaluator.reset()
            cursor = 0
            for i_query, (input, output) in enumerate(zip(inputs_agg, outputs_agg)):
                # creates a fresh copy of the output
                laplacian_output = deepcopy(output)
                # replaces fully connected layer's labels with laplacian's ones
                instances = len(output["instances"].get_fields()["pred_classes"])
                laplacian_output["instances"].set(name="pred_classes",
                                                  value=X_q_labels_laplacian[cursor:cursor + instances])
                cursor += instances
                # evaluates the results
                evaluator.process([input], [laplacian_output])
            evaluation_results = evaluator.evaluate()
            final_results = final_results.append(
                {
                    "use_classification_layer": use_classification_layer,
                    "use_laplacianshot": use_laplacianshot,
                    "rectify_prototypes": rectify_prototypes,
                    "leverage_classification": leverage_classification,
                    "embeddings_type": embeddings_type,
                    **dict(evaluation_results)["bbox"]
                },
                ignore_index=True
            )

        # records the times
        times = times.append(
            {
                "phase": "laplacianshot classification",
                "time_from": int(starting_time),
                "time_to": int(time.time()),
                "time_elapsed": int(time.time() - starting_time)
            },
            ignore_index=True)

    # print("Final results")
    evaluator._logger.info(f"Computation times")
    for column in ["time_from", "time_to"]:
        times[column] = pd.to_datetime(times[column], unit="s")
    print(times[["phase", "time_from", "time_to", "time_elapsed"]])

    evaluator._logger.info(f"Final results")
    print(final_results.sort_values("AP", ascending=False))

    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if evaluation_results is None:
        evaluation_results = {}
    return evaluation_results

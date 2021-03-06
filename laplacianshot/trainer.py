import itertools
import os
import psutil
import time
from datetime import datetime
from os.path import join, splitext, basename, exists, isfile
import logging
from copy import deepcopy
from typing import Optional
from collections import OrderedDict

import numpy as np
import pandas as pd
from compress_pickle import dump, load

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes
from detectron2.utils import comm
from tqdm import tqdm

from fsdet.engine import DefaultTrainer
from fsdet.evaluation import (COCOEvaluator, DatasetEvaluators, LVISEvaluator, PascalVOCDetectionEvaluator,
                              DatasetEvaluator, print_csv_format, inference_context)
from laplacianshot.images_manipulation import normalize_image, apply_random_augmentation
from laplacianshot.inference import laplacian_shot
from laplacianshot.plotting import plot_detections, plot_supports, plot_supports_augmentations, plot_distribution


def get_available_ram_gb():
    memory = psutil.virtual_memory().available * (1024.0 ** -3)
    return memory


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
             support_augmentation: Optional[bool] = True,
             use_laplacianshot: bool = True,
             use_classification_layer: bool = True,
             rectify_prototypes: Optional[bool] = True,
             leverage_classification: Optional[bool] = True,
             embeddings_type: Optional[str] = "embeddings",
             do_pca: Optional[bool] = False,
             remove_possibly_duplicates: Optional[bool] = False,
             knn: Optional[int] = 3,
             lambda_factor: Optional[float] = 0.1,
             max_iters: Optional[int] = None,
             laplacianshot_logs: bool = True,
             plots: bool = False,
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
                                             support_augmentation=support_augmentation,
                                             use_laplacianshot=use_laplacianshot,
                                             use_classification_layer=use_classification_layer,
                                             rectify_prototypes=rectify_prototypes,
                                             leverage_classification=leverage_classification,
                                             embeddings_type=embeddings_type,
                                             do_pca=do_pca,
                                             remove_possibly_duplicates=remove_possibly_duplicates,
                                             knn=knn,
                                             lambda_factor=lambda_factor,
                                             max_iters=max_iters,
                                             cfg=cfg,
                                             save_checkpoints=save_checkpoints,
                                             laplacianshot_logs=laplacianshot_logs,
                                             plots=plots)
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
                         support_augmentation: bool = True,
                         use_laplacianshot: bool = True,
                         use_classification_layer: bool = True,
                         rectify_prototypes: Optional[bool] = True,
                         leverage_classification: Optional[bool] = True,
                         embeddings_type: Optional[str] = "embeddings",
                         do_pca: Optional[bool] = False,
                         remove_possibly_duplicates: Optional[bool] = False,
                         knn: Optional[int] = 3,
                         lambda_factor: Optional[int] = 0.1,
                         max_iters: Optional[int] = None,
                         cfg=None,
                         save_checkpoints: bool = True,
                         plots: bool = False,
                         laplacianshot_logs: bool = True):
    assert not use_laplacianshot or isinstance(use_laplacianshot, bool)
    assert not use_classification_layer or isinstance(use_classification_layer, bool)
    assert not support_augmentation or isinstance(support_augmentation, bool)
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
            original_images_indices = []

            for img_data in tqdm(dataloader_support.dataset.dataset, desc=f"Getting support data"):
                # print(img_data)

                # retrieves data about found image and boxes
                img_original = img_data["image"] \
                    .to(model.device)  # torch.Size([3, H, W])
                boxes_labels = img_data["instances"].get_fields()["gt_classes"] \
                    .to(model.device)  # torch.Size([N])
                boxes_coords = img_data["instances"].get_fields()["gt_boxes"].tensor \
                    .to(model.device)  # torch.Size([N, 4])

                # gt_boxes = torch.zeros_like(boxes_coords)
                gt_boxes = []

                # loops over found boxes
                for i_instance, (box, label) in enumerate(zip(boxes_coords, boxes_labels)):
                    imgs, boxes = [img_original], [box]
                    # tracks non-augmented images
                    original_images_indices += [len(X_s_embeddings)]
                    # eventually data augments the image
                    if support_augmentation is None or support_augmentation:
                        for _ in range(5):
                            img_augmented, box_augmented = apply_random_augmentation(img=img_original, box=box)
                            imgs += [img_augmented]
                            boxes += [box_augmented]

                    for i_img, (img, box) in enumerate(zip(imgs, boxes)):
                        # print(f"Shape and box before: {img.shape}\t\t{box}")
                        # normalizes the image
                        img_normalized = normalize_image(img=img, model=model)
                        # resizes the box according to the new size
                        box_normalized = deepcopy(box)
                        box_normalized[0] = (box_normalized[0] * img_normalized.shape[2]) / img.shape[2]
                        box_normalized[2] = (box_normalized[2] * img_normalized.shape[2]) / img.shape[2]
                        box_normalized[1] = (box_normalized[1] * img_normalized.shape[1]) / img.shape[1]
                        box_normalized[3] = (box_normalized[3] * img_normalized.shape[1]) / img.shape[1]
                        # adjusts img_data
                        img_data_normalized = deepcopy(img_data)
                        img_data_normalized["image"] = (
                                (
                                        (img_normalized + abs(img_normalized.min()))
                                        / img_normalized.max()
                                ) * 255).byte()
                        img_data_normalized["instances"]._image_size = img_normalized.shape[1:]

                        gt_boxes = img_data_normalized["instances"].get("gt_boxes")
                        gt_boxes.tensor[i_instance] = box_normalized
                        img_data_normalized["instances"].set("gt_boxes", [box for box in gt_boxes.tensor])
                        # creates the box proposal query for the classification
                        proposal = [
                            Instances(image_size=img_normalized.shape[-2:],
                                      # objectness_logits=[torch.tensor([1], device=model.device)],
                                      proposal_boxes=Boxes(box_normalized.unsqueeze(0)))
                        ]
                        features = model.backbone(img_normalized.unsqueeze(0))
                        result = model.roi_heads(img_data, features, proposal)[0][0]
                        if len(result.get("box_features")) == 0:
                            continue
                        features = result.get("box_features")[0].type(torch.half)
                        scores = result.get("pred_class_logits")[0].type(torch.half)

                        # keeps relevant infos
                        X_s_imgs += [
                            img[:,
                            int(box[1]): int(box[3]),
                            int(box[0]): int(box[2]),
                            ].cpu()]
                        if X_s_imgs[-1].shape[1] == 0 or X_s_imgs[-1].shape[2] == 0:
                            print(X_s_imgs[-1].shape, img.shape)
                            print(box, box_normalized)
                            exit()
                        X_s_embeddings += [features.cpu()]
                        X_s_probabilities += [scores.cpu()]
                        X_s_labels += [label.cpu()]

            X_s_embeddings, X_s_probabilities, X_s_labels = torch.stack(X_s_embeddings, dim=0), \
                                                            torch.stack(X_s_probabilities, dim=0), \
                                                            torch.stack(X_s_labels, dim=0)

            if plots:
                plot_supports(imgs=[X_s_img for i, X_s_img in enumerate(X_s_imgs)
                                    if i in original_images_indices],
                              labels=X_s_labels[original_images_indices],
                              folder="plots")

            if plots and (support_augmentation is None or support_augmentation):
                plot_supports_augmentations(imgs=X_s_imgs,
                                            labels=X_s_labels,
                                            original_images_indices=original_images_indices,
                                            folder="plots")

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
                if plots and i_query == 0:
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
    if plots:
        plot_distribution(distribution=[
            len(output["instances"].get_fields()["scores"])
            for output in outputs_agg
        ], label_x="detections", title=f"number of detections", folder="plots")

    final_results = pd.DataFrame()

    # evaluates the results using the classification layer
    if use_classification_layer:
        evaluator.reset()
        starting_time = time.time()
        for i_query, (input, output) in enumerate(zip(inputs_agg, outputs_agg)):
            evaluator.process([input], [output])
        evaluation_results = evaluator.evaluate()
        final_results = final_results.append(
            {
                "use_classification_layer": use_classification_layer,
                "use_laplacianshot": use_laplacianshot,
                "time_from": int(starting_time),
                "time_to": int(time.time()),
                "time_elapsed": int(time.time() - starting_time),
                **dict(evaluation_results)["bbox"]
            },
            ignore_index=True
        )

    # evaluates the results using laplacianshot
    if use_laplacianshot:
        combinations = list(itertools.product(
            [True, False] if support_augmentation is None else [support_augmentation],
            [True, False] if rectify_prototypes is None else [rectify_prototypes],
            [True, False] if leverage_classification is None else [leverage_classification],
            [True, False] if remove_possibly_duplicates is None else [remove_possibly_duplicates],
            [True, False] if do_pca is None else [do_pca],
            ["embeddings", "probabilities"] if not embeddings_type else [embeddings_type],
            [1, 3, 5] if not knn else [knn],
            [0.1, 0.3, 0.5, 1.0, 1.5] if not lambda_factor else [lambda_factor]
        ))
        for i_combination, (support_augmentation, rectify_prototypes, leverage_classification,
                            remove_possibly_duplicates, do_pca, embeddings_type, knn, lambda_factor) \
                in enumerate(combinations):
            starting_time = time.time()
            evaluator._logger.info(f"Trying combination {i_combination + 1} out of {len(combinations)} "
                                   f"({np.round(((i_combination + 1) / len(combinations)) * 100, 1)}%)")
            evaluator._logger.info(f"Predicting labels with LaplacianShot using {embeddings_type}")
            evaluator._logger.info(f"support augmentation: {support_augmentation}")
            evaluator._logger.info(f"rectify prototypes: {rectify_prototypes}")
            evaluator._logger.info(f"leverage classification: {leverage_classification}")
            evaluator._logger.info(f"remove possibly duplicates: {remove_possibly_duplicates}")
            evaluator._logger.info(f"do PCA: {do_pca}")
            evaluator._logger.info(f"knn: {knn}")
            evaluator._logger.info(f"lambda factor: {lambda_factor}")
            evaluator._logger.info(f"available RAM: {np.round(get_available_ram_gb(), 3)}GB")
            embeddings_key = "box_features" if embeddings_type == "embeddings" else "pred_class_logits"

            X_s_embeddings_run, X_s_labels_run = X_s_probabilities.detach().clone() \
                                                     if embeddings_type == "probabilities" \
                                                     else X_s_embeddings.detach().clone(), \
                                                 X_s_labels

            if not support_augmentation:
                X_s_embeddings_run, X_s_labels_run = X_s_embeddings_run[original_images_indices], \
                                                     X_s_labels[original_images_indices]

            X_q_ids = [[i] * len(instance_output["instances"])
                       for i, instance_output in enumerate(outputs_agg)]
            X_q_ids = [img_id for instances in X_q_ids for img_id in instances]

            X_q_labels_laplacian = laplacian_shot(
                X_s_embeddings=X_s_embeddings_run,
                X_s_labels=X_s_labels_run,
                X_q_embeddings=torch.stack([field
                                            for instance_output in outputs_agg
                                            for field in instance_output["instances"].get_fields()[embeddings_key]]),
                X_q_labels_pred=torch.stack([field
                                             for instance_output in outputs_agg
                                             for field in instance_output["instances"].get_fields()["pred_classes"]]),
                X_q_pred_confidence=torch.stack([field
                                                 for instance_output in outputs_agg
                                                 for field in instance_output["instances"].get_fields()["scores"]]),
                X_q_probabilties=F.softmax(torch.stack([field
                                                        for instance_output in outputs_agg
                                                        for field in
                                                        instance_output["instances"].get_fields()[
                                                            "pred_class_logits"]]),
                                           dim=-1),
                X_q_ids=torch.as_tensor(X_q_ids),
                proto_rect=rectify_prototypes,
                leverage_classification=leverage_classification,
                embeddings_are_probabilities=True if embeddings_type == "probabilities" else False,
                do_pca=do_pca,
                remove_possibly_duplicates=remove_possibly_duplicates,
                knn=knn, lambda_factor=lambda_factor, logs=laplacianshot_logs, plots=plots)

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
                    "support_augmentation": support_augmentation,
                    "use_classification_layer": use_classification_layer,
                    "use_laplacianshot": use_laplacianshot,
                    "rectify_prototypes": rectify_prototypes,
                    "leverage_classification": leverage_classification,
                    "embeddings_type": embeddings_type,
                    "knn": knn,
                    "remove_possibly_duplicates": remove_possibly_duplicates,
                    "do_pca": do_pca,
                    "lambda_factor": lambda_factor,
                    "time_from": int(starting_time),
                    "time_to": int(time.time()),
                    "time_elapsed": int(time.time() - starting_time),
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

    # =======#=======#=======#=======#=======#=======#=======#=======#=======
    # ======= R E S U L T S
    # =======#=======#=======#=======#=======#=======#=======#=======#=======

    # records the times
    times = times.append(
        {
            "phase": "total time",
            "time_from": times["time_from"].min(),
            "time_to": int(time.time()),
            "time_elapsed": (time.time() - times["time_from"].min())
        },
        ignore_index=True)

    # prints times
    evaluator._logger.info(f"Computation times")
    for column in ["time_from", "time_to"]:
        times[column] = pd.to_datetime(times[column], unit="s")
        final_results[column] = pd.to_datetime(final_results[column], unit="s")
    print(times[["phase", "time_from", "time_to", "time_elapsed"]])

    # prints final results
    evaluator._logger.info(f"Final results")
    final_results = final_results.sort_values("AP", ascending=False)
    pd.set_option('display.max_rows', None)
    print(final_results.drop(columns=["AP50", "AP75", "bAP", "bAP50", "bAP75", "nAP", "nAP50", "nAP75"]))

    # saves results to a .csv
    if not max_iters:
        results_folder = join(".", "results")
        if not exists(results_folder):
            os.makedirs(results_folder)
        dt = datetime.fromtimestamp(time.time())
        filename = f"{dt.strftime('%Y%m%d%H%M%S')}_{'_'.join([model_name, datasets_names])}_results.csv"
        final_results.to_csv(join(results_folder, filename), index=False)
        evaluator._logger.info(f"Results saved to {join(results_folder, filename)}")

    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if evaluation_results is None:
        evaluation_results = {}
    return evaluation_results

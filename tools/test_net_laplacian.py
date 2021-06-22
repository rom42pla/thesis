"""
Detection Testing Script.

This scripts reads a given config file and runs the evaluation.
It is an entry point that is made to evaluate standard models in FsDet.

In order to let one script support evaluation of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use FsDet as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import numpy as np
import torch

from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultTrainer, default_argument_parser, default_setup

import detectron2.utils.comm as comm
import json
import logging
import os
import time
from collections import OrderedDict
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import hooks, launch
from fsdet.evaluation import (
    COCOEvaluator, DatasetEvaluators, LVISEvaluator, PascalVOCDetectionEvaluator, verify_results)

from laplacian import LaplacianTrainer, get_embeddings_and_labels


# class Trainer(DefaultTrainer):
#     """
#     We use the "DefaultTrainer" which contains a number pre-defined logic for
#     standard training workflow. They may not work for you, especially if you
#     are working on a new research project. In that case you can use the cleaner
#     "SimpleTrainer", or write your own training loop.
#     """
#
#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#         """
#         Create evaluator(s) for a given dataset.
#         This uses the special metadata "evaluator_type" associated with each builtin dataset.
#         For your own dataset, you can simply create an evaluator manually in your
#         script and do not have to worry about the hacky if-else logic here.
#         """
#         if output_folder is None:
#             output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
#         evaluator_list = []
#         evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
#         if evaluator_type == "coco":
#             evaluator_list.append(
#                 COCOEvaluator(dataset_name, cfg, True, output_folder)
#             )
#         if evaluator_type == "pascal_voc":
#             return PascalVOCDetectionEvaluator(dataset_name)
#         if evaluator_type == "lvis":
#             return LVISEvaluator(dataset_name, cfg, True, output_folder)
#         if len(evaluator_list) == 0:
#             raise NotImplementedError(
#                 "no Evaluator for the dataset {} with the type {}".format(
#                     dataset_name, evaluator_type
#                 )
#             )
#         if len(evaluator_list) == 1:
#             return evaluator_list[0]
#         return DatasetEvaluators(evaluator_list)
#
#
# class Tester:
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.model = Trainer.build_model(cfg)
#         self.check_pointer = DetectionCheckpointer(
#             self.model, save_dir=cfg.OUTPUT_DIR
#         )
#
#         self.best_res = None
#         self.best_file = None
#         self.all_res = {}
#
#     def test(self, ckpt):
#         self.check_pointer._load_model(self.check_pointer._load_file(ckpt))
#         print("evaluating checkpoint {}".format(ckpt))
#         res = Trainer.test(self.cfg, self.model)
#
#         if comm.is_main_process():
#             verify_results(self.cfg, res)
#             print(res)
#             if (self.best_res is None) or (
#                     self.best_res is not None
#                     and self.best_res["bbox"]["AP"] < res["bbox"]["AP"]
#             ):
#                 self.best_res = res
#                 self.best_file = ckpt
#             print("best results from checkpoint {}".format(self.best_file))
#             print(self.best_res)
#             self.all_res["best_file"] = self.best_file
#             self.all_res["best_res"] = self.best_res
#             self.all_res[ckpt] = res
#             os.makedirs(
#                 os.path.join(self.cfg.OUTPUT_DIR, "inference"), exist_ok=True
#             )
#             with open(
#                     os.path.join(self.cfg.OUTPUT_DIR, "inference", "all_res.json"),
#                     "w",
#             ) as fp:
#                 json.dump(self.all_res, fp)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    model = LaplacianTrainer.build_model(cfg)
    if args.eval_iter != -1:
        # load checkpoint at specified iteration
        ckpt_file = os.path.join(
            cfg.OUTPUT_DIR, "model_{:07d}.pth".format(args.eval_iter - 1)
        )
        resume = False
    else:
        # load checkpoint at last iteration
        ckpt_file = cfg.MODEL.WEIGHTS
        resume = True
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        ckpt_file, resume=resume
    )

    res = LaplacianTrainer.test(cfg, model,
                                max_iters=None)

    if comm.is_main_process():
        verify_results(cfg, res)
        # save evaluation results in json
        os.makedirs(
            os.path.join(cfg.OUTPUT_DIR, "inference"), exist_ok=True
        )
        with open(
                os.path.join(cfg.OUTPUT_DIR, "inference", "res_final.json"),
                "w",
        ) as fp:
            json.dump(res, fp)
    return res


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    if args.eval_during_train or args.eval_all:
        args.dist_url = "tcp://127.0.0.1:{:05d}".format(
            np.random.choice(np.arange(0, 65534))
        )
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

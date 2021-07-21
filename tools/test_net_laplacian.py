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
import random

import numpy as np
import torch

from laplacianshot.trainer import LaplacianTrainer

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import default_argument_parser, default_setup

import detectron2.utils.comm as comm
import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import launch
from fsdet.evaluation import (verify_results)



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
                                use_laplacianshot=False,
                                rectify_prototypes=True,
                                embeddings_type="embeddings",
                                max_iters=None)

    if comm.is_main_process():
        verify_results(cfg, res)
        # save evaluation results in json
        # os.makedirs(
        #     os.path.join(cfg.OUTPUT_DIR, "inference"), exist_ok=True
        # )
        # with open(
        #         os.path.join(cfg.OUTPUT_DIR, "inference", "res_final.json"),
        #         "w",
        # ) as fp:
        #     json.dump(res, fp)
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

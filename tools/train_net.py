# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
import csv
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.engine.hooks import HookBase
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances


# ============================== CUSTOM TRAINER + CSV LOGGER ==============================

class CSVLoggerHook(HookBase):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.fields = ['iteration', 'total_loss', 'validation_AP']  # ✅ "time" 제거
        self.best_ap = 0.0
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

    def after_step(self):
        storage = self.trainer.storage
        row = {
            'iteration': self.trainer.iter,
            'total_loss': storage.history("total_loss").latest() or 0.0,
            'validation_AP': getattr(self.trainer, "last_ap", -1),
        }
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(row)



class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            ))
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(f"No Evaluator for dataset {dataset_name} with type {evaluator_type}")
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        eval_hook = hooks.EvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            lambda: self.test_and_track_ap()
        )
        csv_logger = CSVLoggerHook(os.path.join(self.cfg.OUTPUT_DIR, "metrics_log.csv"))
        return hooks_list + [eval_hook, csv_logger]

    def test_and_track_ap(self):
        results = self.test(self.cfg, self.model)
        ap = results["bbox"]["AP"]
        self.last_ap = ap
        if ap > getattr(self, 'best_ap', 0):
            self.best_ap = ap
            torch.save(self.model.state_dict(), os.path.join(self.cfg.OUTPUT_DIR, "best_model.pth"))
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        return OrderedDict({k + "_TTA": v for k, v in res.items()})


# ============================== SETUP & MAIN ==============================

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    register_coco_instances("icdar2013_train", {}, "input_images/annotations/train.json", "input_images/train")
    register_coco_instances("icdar2013_test", {}, "input_images/annotations/test.json", "input_images"
    "/test")


    cfg = setup(args)

    if args.eval_only:
        model = CustomTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = CustomTrainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(CustomTrainer.test_with_TTA(cfg, model))
        return res

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks([hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))])
    return trainer.train()


# ============================== ENTRY POINT ==============================

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

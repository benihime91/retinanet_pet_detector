import argparse
from typing import Dict, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from pytorch_retinanet.references import (
    CocoEvaluator,
    DetectionDataset,
    get_coco_api_from_dataset,
)
from pytorch_retinanet.retinanet.utilities import collate_fn

from .utils import get_tfms, load_obj


class DetectionModel(pl.LightningModule):
    def __init__(self, model: nn.Module, hparams: Union[Dict, argparse.Namespace, DictConfig]):
        super(DetectionModel, self).__init__()
        self.model = model
        self.hparams = hparams

    # ===================================================== #
    # Configure the Optimizer & Scheduler for the Model
    # ===================================================== #
    def configure_optimizers(self, *args, **kwargs):
        "instatiates optimizer & scheduler(s)"
        params = [p for p in self.model.parameters() if p.requires_grad]
        # optimizer
        optimizer = load_obj(self.hparams.optimizer.class_name)(
            params, **self.hparams.optimizer.params
        )
        # scheduler
        scheduler = load_obj(self.hparams.scheduler.class_name)(
            optimizer, **self.hparams.scheduler.params
        )

        return [optimizer], [scheduler]

    # ===================================================== #
    # Getting the data ready
    # ===================================================== #
    def prepare_data(self, stage=None):
        """
        load in the transformation & reads in the data from given paths.
        """
        # instantiate the transforms
        self.tfms = get_tfms(self.hparams)
        # load in the csv files
        self.trn_df = pd.read_csv(self.hparams.train_csv)
        self.val_df = pd.read_csv(self.hparams.valid_csv)
        self.test_df = pd.read_csv(self.hparams.test_csv)

    # ===================================================== #
    # Forward pass of the Model
    # ===================================================== #
    def forward(self, xb, *args, **kwargs):
        "forward step"
        return self.model(xb)

    # ===================================================== #
    # Training
    # ===================================================== #
    def train_dataloader(self, *args, **kwargs):
        "instantiate train dataloader"
        # instantiate the trian dataset
        train_ds = DetectionDataset(self.trn_df, self.tfms["train"])
        # load in the dataloader
        bs = self.hparams.train_batch_size
        trn_dl = DataLoader(train_ds, bs, True, collate_fn=collate_fn, pin_memory=True,)
        return trn_dl

    def training_step(self, batch, batch_idx, *args, **kwargs):
        "one training step"
        images, targets, _ = batch  # unpack the one batch from the DataLoader
        targets = [{k: v for k, v in t.items()} for t in targets]  # Unpack the Targets
        # Calculate Losses {regression_loss , classification_loss}
        loss_dict = self.model(images, targets)
        # Calculate Total Loss
        losses = sum(loss for loss in loss_dict.values())
        return {"loss": losses, "log": loss_dict, "progress_bar": loss_dict}

    # ===================================================== #
    # Validation
    # ===================================================== #
    def val_dataloader(self, *args, **kwargs):
        "instatiate validation dataloader"
        # instantiate the validaiton dataset
        val_ds = DetectionDataset(self.val_df, self.tfms["valid"])
        # instantiate dataloader
        bs = self.hparams.valid_batch_size
        loader = DataLoader(val_ds, bs, collate_fn=collate_fn,)
        # instantiate coco
        coco = get_coco_api_from_dataset(loader.dataset)
        self.coco_evaluator = CocoEvaluator(coco, [self.hparams.iou_types])
        return loader

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        "one validation step"
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images, targets)
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }
        self.coco_evaluator.update(res)
        return {}

    def validation_epoch_end(self, outputs, *args, **kwargs):
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()
        metric = self.coco_evaluator.coco_eval["bbox"].stats[0]
        metric = torch.as_tensor(metric)
        logs = {"valid_mAP": metric}

        return {
            "valid_mAP": metric,
            "log": logs,
            "progress_bar": logs,
        }

    # ===================================================== #
    # Test
    # ===================================================== #
    def test_dataloader(self, *args, **kwargs):
        "instatiate validation dataloader"
        # instantiate train dataset
        test_ds = DetectionDataset(self.test_df, self.tfms["test"])
        # instantiate dataloader
        bs = self.hparams.test_batch_size
        loader = DataLoader(test_ds, bs, collate_fn=collate_fn,)
        # instantiate coco_api to track metrics
        coco = get_coco_api_from_dataset(loader.dataset)
        self.test_evaluator = CocoEvaluator(coco, [self.hparams.iou_types])
        return loader

    def test_step(self, batch, batch_idx, *args, **kwargs):
        "one test step"
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images, targets)
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }
        self.test_evaluator.update(res)
        return {}

    def test_epoch_end(self, outputs, *args, **kwargs):
        self.test_evaluator.accumulate()
        self.test_evaluator.summarize()
        metric = self.test_evaluator.coco_eval["bbox"].stats[0]
        metric = torch.as_tensor(metric)
        logs = {"test_mAP": metric}

        return {
            "test_mAP": metric,
            "log": logs,
            "progress_bar": logs,
        }

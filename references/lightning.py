import argparse
from typing import Dict, Union

import pytorch_lightning as pl
from torch import nn

from ..pytorch_retinanet.references import (CocoEvaluator, get_coco_api_from_dataset)
from ..pytorch_retinanet.references.dataset import DetectionDataset
from ..pytorch_retinanet.retinanet.utilities import collate_fn


class DetectionModel(pl.LightningModule):
    def __init__(self, model: nn.Module, hparams: Union[Dict, argparse.Namespace, Dict]):
        super(DetectionModel, self).__init__()
        self.model = model
        self.hparams = hparams

    # ===================================================== #
    # Configure the Optimizer & Scheduler for the Model
    # ===================================================== #
    def configure_optimizers(self, *args, **kwargs):
        "instatiates optimizer & scheduler(s)"
        # optimizer
        optimizer = self.hparams.optimizer
        # scheduler
        scheduler = self.hparams.scheduler
        if scheduler is not None:
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    # ===================================================== #
    # Getting the data ready
    # ===================================================== #
    def prepare_data(self, stage=None):
        """
        load in the transformation & reads in the data from given paths.
        """
        self.tfms = get_tfms()
        self.trn_df = pd.read_csv(self.hparams.trn_df)  
        self.val_df = pd.read_csv(self.hparams.val_df)  
        self.test_df = pd.read_csv(self.hparams.test_df) 
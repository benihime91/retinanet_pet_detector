import os
from typing import Dict, Union
import datetime

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateLogger,
    ModelCheckpoint,
)
from torch import nn
from torch.utils.data import DataLoader

from pytorch_retinanet.utils import CocoEvaluator, get_coco_api_from_dataset
from pytorch_retinanet import DetectionDataset
from pytorch_retinanet.retinanet.utilities import collate_fn

from .data_utils import _get_logger
from .utils import get_tfms, load_obj


class DetectionModel(pl.LightningModule):
    """
    Lightning Class to wrap the RetinaNet Model.
    So that it can be trainer with LightningTrainer.

    Args:
      model (`nn.Module`)    : A `RetinaNet` model instance. 
      haprams (`DictConfig`) : A `DictConfig` that stores the configs for training .
                               Check `main.yaml` in the parent dir.
    """

    def __init__(self, model: nn.Module, hparams: DictConfig):
        super(DetectionModel, self).__init__()
        self.model = model
        self.hparams = hparams
        # instantiate logger
        self.fancy_logger = _get_logger(__name__)

    # ===================================================== #
    # Configure the Optimizer & Scheduler for the Model
    # ===================================================== #
    def configure_optimizers(self, *args, **kwargs):
        "instatiates optimizer & scheduler(s)"
        params = [p for p in self.model.parameters() if p.requires_grad]
        # intialize optimizer
        self.optimizer = load_obj(self.hparams.optimizer.class_name)(
            params, **self.hparams.optimizer.params
        )

        # initialize scheduler
        self.scheduler = load_obj(self.hparams.scheduler.class_name)(
            self.optimizer, **self.hparams.scheduler.params
        )
        self.scheduler = {
            "scheduler": self.scheduler,
            "interval": self.hparams.scheduler.interval,
            "frequency": self.hparams.scheduler.frequency,
        }

        # log optimizer and scheduler
        self.fancy_logger.info(f"OPTIMIZER_NAME : {self.optimizer.__class__.__name__}")
        self.fancy_logger.info(f"LEARNING_RATE: {self.hparams.optimizer.params.lr}")
        self.fancy_logger.info(f"WEIGHT_DECAY: {self.hparams.optimizer.params.weight_decay}")
        self.fancy_logger.info(f"LR_SCHEDULER_NAME : {self.scheduler['scheduler'].__class__.__name__}")
        return [self.optimizer], [self.scheduler]

    # ===================================================== #
    # Getting the data ready
    # ===================================================== #
    def prepare_data(self, stage=None):
        """
        load in the transformation & reads in the data from given paths.
        """
        # instantiate the transforms
        self.tfms = get_tfms(self.hparams)
        _augs = self.tfms["train"].transforms
        prompt = [_augs[i].__class__.__name__ for i in range(len(list(_augs)))]
        self.fancy_logger.info(f"Augmentations used in training: {prompt}")

        # load in the csv files
        # train csv
        self.trn_df = pd.read_csv(self.hparams.train_csv)
        self.fancy_logger.info(f"Loaded train dataset from {self.hparams.train_csv}")
        self.fancy_logger.info(f"Loaded dataset takes {os.path.getsize(self.hparams.train_csv)/(1024*1024):.2f} MiB")
        # validation csv file
        self.val_df = pd.read_csv(self.hparams.valid_csv)
        self.fancy_logger.info(f"Loaded validation dataset from {self.hparams.valid_csv}")
        self.fancy_logger.info(f"Loaded dataset takes {os.path.getsize(self.hparams.valid_csv)/(1024*1024):.2f} MiB")
        # test csv file
        self.test_df = pd.read_csv(self.hparams.test_csv)
        self.fancy_logger.info(f"Loaded test dataset from {self.hparams.test_csv}")
        self.fancy_logger.info(f"Loaded dataset takes {os.path.getsize(self.hparams.test_csv)/(1024*1024):.2f} MiB")

    # ===================================================== #
    # Forward pass of the Model
    # ===================================================== #
    def forward(self, xb, *args, **kwargs):
        return self.model(xb)

    # ===================================================== #
    # Training
    # ===================================================== #
    def train_dataloader(self, *args, **kwargs):
        # instantiate the trian dataset
        train_ds = DetectionDataset(self.trn_df, self.tfms["train"])
        # load in the dataloader
        bs = self.hparams.train_batch_size
        trn_dl = DataLoader(train_ds, bs, True, collate_fn=collate_fn, **self.hparams.dataloader)
        return trn_dl

    def training_step(self, batch, batch_idx, *args, **kwargs):
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
        # instantiate the validaiton dataset
        val_ds = DetectionDataset(self.val_df, self.tfms["valid"])
        # instantiate dataloader
        bs = self.hparams.valid_batch_size
        loader = DataLoader(val_ds, bs, collate_fn=collate_fn, **self.hparams.dataloader)
        return loader

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, targets, _ = batch  # unpack the one batch from the DataLoader
        targets = [{k: v for k, v in t.items()} for t in targets]  # Unpack the Targets
        # Calculate Losses {regression_loss , classification_loss}
        loss_dict = self.model(images, targets)
        # Calculate Total Loss
        loss = sum(loss for loss in loss_dict.values())
        loss = torch.as_tensor(loss)
        logs = {"val_loss": loss}
        return {"val_loss": loss, "log": logs, "progress_bar": logs,}

    # ===================================================== #
    # Test
    # ===================================================== #
    def test_dataloader(self, *args, **kwargs):
        # instantiate train dataset
        test_ds = DetectionDataset(self.test_df, self.tfms["test"])
        # instantiate dataloader
        bs = self.hparams.test_batch_size
        loader = DataLoader(test_ds, bs, collate_fn=collate_fn, **self.hparams.dataloader)
        prompt = "Converting dataset annotations in 'test_dataset' to COCO format ..."
        self.fancy_logger.info(prompt)

        # instantiate coco_api to track metrics
        coco = get_coco_api_from_dataset(loader.dataset)
        self.test_evaluator = CocoEvaluator(coco, [self.hparams.iou_types])
        prompt = f"Conversion finished, num images: {loader.dataset.__len__()}"
        self.fancy_logger.info(prompt)
        return loader

    def test_step(self, batch, batch_idx, *args, **kwargs):
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model.predict(images)
        res = {t["image_id"].item(): o for t, o in zip(targets, outputs)}
        self.test_evaluator.update(res)
        return {}

    def test_epoch_end(self, outputs, *args, **kwargs):
        # coco results
        self.fancy_logger.info("Preparing results for COCO format ...")
        self.fancy_logger.info("Evaluating predictions ...")
        self.test_evaluator.accumulate()
        self.test_evaluator.summarize()
        metric = self.test_evaluator.coco_eval["bbox"].stats[0]
        metric = torch.as_tensor(metric)
        logs = {"AP": metric}
        return {"AP": metric, "log": logs, "progress_bar": logs,}


class LogCallback(pl.Callback):
    """
    Callback to handle logging within pl_module
    """
    def __init__(self, cfg: Union[DictConfig, Dict]) -> None:
        self.cfg = cfg

    def on_fit_start(self, trainer, pl_module):
        trn_bs = pl_module.hparams.train_batch_size
        val_bs = pl_module.hparams.valid_batch_size
        tst_bs = pl_module.hparams.test_batch_size
        output_dir = self.cfg.model_checkpoint.params.filepath
        eps = self.cfg.flags.max_epochs
        
        pl_module.fancy_logger.info(f"IMS_PER_TRAIN_BATCH : {trn_bs}")
        pl_module.fancy_logger.info(f"IMS_PER_VALIDATION_BATCH : {val_bs}")
        pl_module.fancy_logger.info(f"IMS_PER_TEST_BATCH : {tst_bs}")
        pl_module.fancy_logger.info(f"CHECKPOINT_DIR : {output_dir}")
        pl_module.fancy_logger.info(f"MAX_EPOCHS : {eps}")

    def on_train_start(self, trainer, pl_module):
        self.train_start = datetime.datetime.now().replace(microsecond=0)
        prompt = f"Training on {pl_module.train_dataloader().dataset.__len__()} images"
        pl_module.fancy_logger.info(prompt)
        prompt = f"Training from iteration {trainer.global_step} : "
        pl_module.fancy_logger.info(prompt)

    def on_train_end(self, trainer, pl_module):
        self.train_end = datetime.datetime.now().replace(microsecond=0)
        prompt = f" Total compute time : {self.train_end - self.train_start}"
        pl_module.fancy_logger.info(prompt)
        
    def on_test_start(self, trainer, pl_module):
        self.test_start = datetime.datetime.now().replace(microsecond=0)
        prompt = f"Start Inference on {pl_module.test_dataloader().dataset.__len__()} images"
        pl_module.fancy_logger.info(prompt)

    def on_test_end(self, trainer, pl_module):
        self.test_end = datetime.datetime.now().replace(microsecond=0)
        prompt = f" Total inference time : {self.test_end - self.test_start}"
        pl_module.fancy_logger.info(prompt)





def initialize_trainer(trainer_conf: Union[DictConfig, Dict], **kwargs) -> pl.Trainer:
    """
    Instantiates a Lightning Trainer from given config file .
    The Trainer is initialized with the flags given in the config
    file with the addition of the `LogCallback`.

    Args:
        trainer_conf `(DictConfig)`: configs for the Trainer.
        **kwargs – Other arguments are passed directly to the `pl.Trainer`.
    """
    # instantiate EarlyStoppping Callback
    early_stopping = EarlyStopping(**trainer_conf.early_stopping.params)

    # instantiate ModelCheckpoint Callback
    os.makedirs(trainer_conf.model_checkpoint.params.filepath, exist_ok=True)
    model_checkpoint = ModelCheckpoint(**trainer_conf.model_checkpoint.params)

    # instantiate callbacks
    lr_logger = LearningRateLogger(**trainer_conf.learning_rate_monitor.params)
    logger = load_obj(trainer_conf.logger.class_name)(**trainer_conf.logger.params)
    log_cb = LogCallback(cfg=trainer_conf)

    callbacks = [lr_logger, log_cb]
    logger = [logger]

    # Load Trainer:
    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=model_checkpoint,
        early_stop_callback=early_stopping,
        callbacks=callbacks,
        **trainer_conf.flags,
        **kwargs,
    )

    return trainer

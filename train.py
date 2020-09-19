import argparse
import datetime
import logging
import os
import warnings

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from pytorch_retinanet.retinanet.models import Retinanet
from references import DetectionModel, initialize_trainer
from references.data_utils import _get_logger


def main(args: argparse.Namespace, seed: int = 123):
    logger = _get_logger(name=__name__)

    # set lightning seed to results are reproducible
    pl.seed_everything(seed)
    logger.name = "Lightning"
    logger.info(f"Random seed = {seed}")

    # load the config file
    cfg = OmegaConf.load(args.config)
    if args.disp:
        logger.name = "main.yaml"
        logger.info(f"[Configurations]: \n {OmegaConf.to_yaml(cfg)}")

    # instantiate Retinanet model
    logger.name = "pytorch_retinanet.retinanet.models"
    model = Retinanet(**cfg.model, logger=logger)
    logger.info(f"Model: \n {model}")

    logger.name = "retinanet_pet_detector"
    # Instantiate LightningModel & Trainer
    litModule = DetectionModel(model, cfg.hparams)
    trainer = initialize_trainer(cfg.trainer, weights_summary=None)

    # Train and validation
    strt_time = datetime.datetime.now()
    logger.info(f"Starting training from iteration {0}")
    trainer.fit(litModule)
    end_tim = datetime.datetime.now()
    tot = end_tim - strt_time
    logger.info(f"Total training time:  {tot}")

    # Test
    logger.info("Evaluation results for bbox on test set: ")
    strt_time = datetime.datetime.now()
    trainer.test(litModule)
    end_tim = datetime.datetime.now()
    tot = end_tim - strt_time
    logger.info(f"Total inference time:  {tot}")

    # Save weights
    logger.info("Serializing model weights .... ")
    weights = os.path.join(cfg.trainer.model_checkpoint.params.filepath, "weights.pt")
    torch.save(litModule.model.state_dict(), weights)
    logger.info(f"Weights saved to {weights} .... ")


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, type=str, help="path to the config file"
    )
    parser.add_argument(
        "--disp",
        required=False,
        default=True,
        help="wether to print out the config",
        type=bool,
    )

    arguments = parser.parse_args()
    main(args=arguments)

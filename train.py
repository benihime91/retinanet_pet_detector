import argparse
import datetime
import logging
import os
import sys
import warnings

import torch
from omegaconf import OmegaConf
from termcolor import colored

from pytorch_retinanet.retinanet.models import Retinanet
from pytorch_retinanet.retinanet.utilities import ifnone
from references import DetectionModel, initialize_trainer
from references.data_utils import _ColorfulFormatter, _get_logger


def main(args: argparse.Namespace, logger: logging.Logger):
    import pytorch_lightning as pl

    # set lightning seed to results are reproducible
    seed = 123
    pl.seed_everything(123)
    logger.name = "lightning"
    logger.info(f"Random seed = {seed}")

    # load the config file
    cfg = OmegaConf.load(args.config)
    if args.disp:
        logger.name = "configurations"
        logger.info("Configs:")
        print(OmegaConf.to_yaml(cfg))

    # instantiate Retinanet model
    logger.name = "retinanet"
    model = Retinanet(**cfg.model, logger=logger)
    logger.info(f"Model: \n {model}")

    logger.name = "pet-detector"
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
    logger = _get_logger()

    main(args=arguments, logger=logger)

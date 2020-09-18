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


def main(args, logger):
    cfg = OmegaConf.load(args.config)
    if args.disp:
        logger.info("Config:")
        print(OmegaConf.to_yaml(cfg))

    # instantiate Retinanet model
    model = Retinanet(**cfg.model, logger=logger)
    logger.info("Model: ")
    print(model)

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
    weights = os.path.join(cfg.trainer.model_checkpoint.filepath, "weights.pt")
    torch.save(litModule.model.state_dict(), weights)
    logger.info(f"Weights saved to {weights} .... ")


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def _get_logger(name=None):
    # Set up Logging
    name = ifnone(name, "retinanet")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    logging.basicConfig(
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.DEBUG,
    )
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = _ColorfulFormatter(
        colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
        datefmt="%m/%d %H:%M:%S",
        root_name="retinanet",
        abbrev_name=str("retinanet"),
    )

    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


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

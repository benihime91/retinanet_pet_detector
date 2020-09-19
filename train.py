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
    """
    Fn to initialize the LitniningModul, LightningTrainer
    for train , validation & evaluation.
    Training, validaiton and evaluation is controled by a .yaml config file
    To see an exmaple config file see `main.yaml`. 
    To config file can be modified by adding or modifying the existing arguments.

    Args:
        1. args (argparse.Namespace): args shuld contain 2 items.
            - config (str)   : path to the config file.
            - verbose (int)  : verbosity if verbose is > 0 prints out all the model
                               arch and the configuration file. Verbose 0 prints out
                               the logs during train, validaiton and test
        
        2. seed (int) : A number to seed lightning seed. Lightning seed ensures that our model 
                     results are reproducible.
    """
    logger = _get_logger(name=__name__)

    # set lightning seed to results are reproducible
    pl.seed_everything(seed)
    logger.name = "Lightning"
    logger.info(f"Random seed = {seed}")

    # load the config file
    cfg = OmegaConf.load(args.config)
    # if versbose > 0 : print out the config file arguments
    if args.verbose > 0:
        logger.name = "main.yaml"
        logger.info(f"[Configurations]: \n {OmegaConf.to_yaml(cfg)}")

    #  set logger name
    logger.name = "pytorch_retinanet.retinanet.models" 
    # Instantiate Retinanet model
    model = Retinanet(**cfg.model, logger=logger)
    # print logs
    logger.info(f"Image Resize parameters: smallest_image_size={cfg.model.min_size}")
    logger.info(f"Image Resize parameters: maximum_image_size={cfg.model.max_size}")

    if args.verbose > 0:
        logger.info(f"Model: \n {model}")

    logger.name = "retinanet_pet_detector"
    
    # Instantiate LightningModel & Trainer
    litModule = DetectionModel(model, cfg.hparams)
    trainer = initialize_trainer(cfg.trainer, weights_summary=None)

    # Train and validation
    strt_time_1 = datetime.datetime.now()
    trainer.fit(litModule)
    end_tim = datetime.datetime.now()
    tot = end_tim - strt_time_1
    logger.info(f"Total training time:  {tot}")

    # Test
    strt_time = datetime.datetime.now()
    trainer.test(litModule)
    end_tim = datetime.datetime.now()
    tot = end_tim - strt_time
    logger.info(f"Total inference time:  {tot}")

    # Save weights
    weights = os.path.join(cfg.trainer.model_checkpoint.params.filepath, "weights.pt")
    torch.save(litModule.model.state_dict(), weights)
    logger.info("serializing model state dict ...")
    logger.info(f"Weights saved to {weights} .... ")
    logger.info("Cleaning up .....")
    t_end = datetime.datetime.now()
    logger.info(f"Overall time elasped : {strt_time_1 - t_end}")


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, type=str, help="path to the config file"
    )
    parser.add_argument(
        "--verbose",
        required=False,
        default=0,
        help="wether to print out the config and model",
        type=int,
    )

    arguments = parser.parse_args()
    main(args=arguments)

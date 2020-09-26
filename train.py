import argparse
import os
from typing import Dict, Union

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig

from pytorch_retinanet.retinanet.models import Retinanet
from references import DetectionModel, initialize_trainer
from references.data_utils import _get_logger


def main(args: Union[argparse.Namespace, DictConfig, Dict], seed: int = 123):
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
    logger.info(f"SEED: {seed}")
    # load the config file
    cfg = OmegaConf.load(args.config)
    
    # if versbose > 0 : print out the config file arguments
    if args.verbose > 0:        
        logger.info(f"Contents of args.config = {args.config}: \n {OmegaConf.to_yaml(cfg)}")

    # Instantiate Retinanet model
    model = Retinanet(**cfg.model, logger=logger)
    logger.info(f"INPUT_PARAMS: MIN_IMAGE_SIZE = {cfg.model.min_size}\tMAX_IMAGE_SIZE = {cfg.model.max_size}")

    if args.verbose > 0:        
        logger.info(f"Model: \n {model}")
    
    # Instantiate LightningModel & Trainer
    litModule = DetectionModel(model, cfg.hparams)
    trainer = initialize_trainer(cfg.trainer, weights_summary=None)
    # Train and validation
    trainer.fit(litModule)
    # Test
    trainer.test(litModule)
    # Save weights
    weights = os.path.join(cfg.trainer.model_checkpoint.params.filepath, "weights.pth")
    # NB: use_new_zipfile_serialization = True causes problems while loading the model
    torch.save(litModule.model.state_dict(), weights, _use_new_zipfile_serialization=False)
    logger.info("Saving model weights ...")
    print(f"## Model weights saved to {weights}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="path to the config file")
    parser.add_argument("--verbose", required=False, default=0, help="wether to print out the config and model",type=int,)
    arguments = parser.parse_args()
    main(args=arguments)

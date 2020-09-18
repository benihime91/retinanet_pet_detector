import argparse
import ast
import importlib
from typing import *

import albumentations as A
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from PIL import Image
from torch import nn

from pytorch_retinanet.retinanet.models import Retinanet
from .display_preds import Visualizer

# Path to the Label Dictionary
LABEL_PATH = "labels.names"


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def get_tfms(conf: DictConfig) -> Dict[str, A.Compose]:
    """
    Loads in albumentation augmentations for train, valid, test
    from given config as a dictionary.
    """
    trn_tfms = [
        load_obj(i["class_name"])(**i["params"]) for i in conf.augmentation.train
    ]
    val_tfms = [
        load_obj(i["class_name"])(**i["params"]) for i in conf.augmentation.valid
    ]
    test_tfms = [
        load_obj(i["class_name"])(**i["params"]) for i in conf.augmentation.test
    ]

    # transforms dictionary :
    transforms = {
        "train": A.Compose(
            trn_tfms,
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels"]
            ),
        ),
        "valid": A.Compose(
            val_tfms,
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels"]
            ),
        ),
        "test": A.Compose(
            test_tfms,
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels"]
            ),
        ),
    }

    return transforms


def get_label_dict(fname):
    # read in the Label Dictionary
    f = open(fname, "r")
    label_dict = f.read()
    label_dict = ast.literal_eval(label_dict)
    f.close()
    return label_dict


# Dictionary conating a mapping of the Labels
label_dict = get_label_dict(LABEL_PATH)

transforms = A.Compose([A.ToFloat(), ToTensorV2()])


def load_yaml_config(path) -> Dict:
    "load a yaml config and returns a dictionary"
    with open(path, "r") as f:
        conf_dict = yaml.load(f, Loader=yaml.FullLoader)
    return conf_dict


def get_model(args: argparse.Namespace):
    "returns a pre-trained retinanet model"
    model = Retinanet(
        args.num_classes,
        args.model_backbone,
        score_thres=args.score_thres,
        nms_thres=args.nms_thres,
    )
    state_dict = torch.hub.load_state_dict_from_url(args.url, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def get_preds(model: nn.Module, image: Union[np.array, str]) -> Tuple[List]:
    """
    Generated predictions for the given `image` using `model`.
    """
    import logging

    logger = logging.getLogger(__name__)

    # load in the image if string is give
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
        # Convert PIL image to array
        image = np.array(image)

    # Convert Image to a tensor
    tensor_image = transforms(image=image)["image"]
    # Generate predicitons
    model.eval()
    pred = model([tensor_image])

    # Gather the bbox, scores & labels from the preds
    pred_boxes = pred[0]["boxes"]  # Bounding boxes
    pred_class = pred[0]["labels"]  # predicted class labels
    pred_score = pred[0]["scores"]  # predicted scores

    # Process detections
    boxes = list(pred_boxes.cpu().numpy())
    clas = list(pred_class.cpu().numpy())
    scores = list(pred_score.cpu().numpy())

    logger.info(f"Number of bounding doxes detected : {len(boxes)}")

    return boxes, clas, scores


def detection_api(
    model: torch.nn.Module,
    img: str,
    save: bool = True,
    show: bool = False,
    fname: str = "res.png",
    save_dir: str = "outputs",
):

    """
    Generates & draws the bbox predictions over the given image
    using the given retinanet model.
    
    Args:
     model (nn.Module): An object detection model which gives bbounding box predictions for given image.
     img    (str)     : path to the image on which predicitons are to be made.
     save   (bool)    : wether to save the generated image.
     show   (bool)    : wether to display the generated result.
     fname  (str)     : filename of the generated image.
     save_dir (str)   : directory where to save the image.
    """
    # Extract the predicitons for given Image
    bb, cls, sc = get_preds(model, img)
    # Instantiate the visualizer
    viz = Visualizer(class_names=label_dict)
    # Load in the image
    img = Image.open(img).convert("RGB")
    img = np.array(img)
    # Draw the bounding boxes over the loaded image
    viz.draw_bboxes(
        img, bb, cls, sc, save=save, show=show, save_dir=save_dir, fname=fname
    )

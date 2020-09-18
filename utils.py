import argparse
import ast
from typing import *

import albumentations as A
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch import nn

from display_preds import Visualizer
from pytorch_retinanet.retinanet.models import Retinanet

# read in the Label Dictionary
f = open("labels.names", "r")
label_dict = f.read()
label_dict = ast.literal_eval(label_dict)
f.close()

# Instantiate the visualizer
viz = Visualizer(class_names=label_dict)

transforms = A.Compose([A.ToFloat(), ToTensorV2()])


def load_yaml_config(path) -> Dict:
    "load a yaml config and returns a dictionary"
    with open(path, "r") as f:
        conf_dict = yaml.load(f, Loader=yaml.FullLoader)
    return conf_dict


def get_model(args: argparse.Namespace):
    "returns a pre-trained retinanet model"
    model = Retinanet(
        num_classes=args.num_classes,
        backbone_kind=args.model_backbone,
        score_thres=args.score_thres,
        nms_thres=args.nms_thres,
        max_detections_per_images=args.max_detections,
    )

    state_dict = torch.hub.load_state_dict_from_url(args.url, map_location="cpu")

    model.load_state_dict(state_dict)

    return model


@torch.no_grad()
def get_preds(model: nn.Module, image: Union[np.array, str]) -> Tuple[List]:
    "get predictions for the uploaded image"
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

    return boxes, clas, scores


def detection_api(
    model: torch.nn.Module,
    img: str,
    save: bool = True,
    show: bool = False,
    fname: str = "res.png",
    save_dir: str = "outputs",
) -> None:

    "Draw bbox predictions on given image"
    # Extract the predicitons for given Image
    bb, cls, sc = get_preds(model, img)

    # Draw bounding boxes
    img = Image.open(img).convert("RGB")
    img = np.array(img)

    viz.draw_bboxes(
        img,
        boxes=bb,
        classes=cls,
        scores=sc,
        save=save,
        show=show,
        save_dir=save_dir,
        fname=fname,
    )

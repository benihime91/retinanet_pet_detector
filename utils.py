import argparse
from typing import *

import albumentations as A
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch import nn

from display_preds import Visualizer
from pytorch_retinanet.src.models import Retinanet

label_dict = {
    0: "abyssinian",
    1: "american_bulldog",
    2: "american_pit_bull_terrier",
    3: "basset_hound",
    4: "beagle",
    5: "bengal",
    6: "birman",
    7: "bombay",
    8: "boxer",
    9: "british_shorthair",
    10: "chihuahua",
    11: "egyptian_mau",
    12: "english_cocker_spaniel",
    13: "english_setter",
    14: "german_shorthaired",
    15: "great_pyrenees",
    16: "havanese",
    17: "japanese_chin",
    18: "keeshond",
    19: "leonberger",
    20: "maine_coon",
    21: "miniature_pinscher",
    22: "newfoundland",
    23: "persian",
    24: "pomeranian",
    25: "pug",
    26: "ragdoll",
    27: "russian_blue",
    28: "saint_bernard",
    29: "samoyed",
    30: "scottish_terrier",
    31: "shiba_inu",
    32: "siamese",
    33: "sphynx",
    34: "staffordshire_bull_terrier",
    35: "wheaten_terrier",
    36: "yorkshire_terrier",
}

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
    print("[INFO] Generating Predictions ..... ")

    bb, cls, sc = get_preds(model, img)

    print(f"[INFO] {len(bb)} bounding_boxes detected ....")
    print("[INFO] creating bbox on the image .... ")

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

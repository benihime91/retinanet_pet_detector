import argparse
from typing import *

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch import nn
from torchvision.ops.boxes import batched_nms

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


def get_model(args: argparse.Namespace, **kwargs):
    "returns a pre-trained retinanet model"
    model = Retinanet(
        num_classes=args.num_classes, backbone_kind=args.model_backbone, **kwargs
    )
    # if load model from url
    if args.load_from_url:
        state_dict = state_dict = torch.hub.load_state_dict_from_url(
            args.url, map_location="cpu"
        )
        model.load_state_dict(state_dict)
    else:
        # else load model from given path
        model.load_state_dict(torch.load(args.state_dict_path))
    return model


@torch.no_grad()
def get_preds(
    model: torch.nn.Module,
    path: str,
    threshold: float,
    iou_threshold: float,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[List, List, List]:
    "Get predictions on image"

    model.to(device)
    # Load the imag
    img = Image.open(path).convert("RGB")
    img = np.array(img)

    # Process the image
    img = transforms(image=img)["image"]
    img.to(device)

    # Generate predictions
    model.eval()
    pred = model([img])

    # Gather the bbox, scores & labels from the preds
    pred_boxes = pred[0]["boxes"]  # Bounding boxes
    pred_class = pred[0]["labels"]  # predicted class labels
    pred_score = pred[0]["scores"]  # predicted scores

    # Get list of index with score greater than threshold.
    mask = pred_score > threshold

    # Filter predictions
    boxes = pred_boxes[mask]
    clas = pred_class[mask]
    scores = pred_score[mask]

    # do NMS
    keep_idxs = batched_nms(boxes, scores, clas, iou_threshold)
    boxes = list(boxes[keep_idxs].cpu().numpy())
    clas = list(clas[keep_idxs].cpu().numpy())
    scores = list(scores[keep_idxs].cpu().numpy())

    return boxes, clas, scores


def detection_api(
    model: torch.nn.Module,
    img: str,
    score_thres: float = 0.5,
    iou_thres: float = 0.2,
    save: bool = True,
    show: bool = False,
    fname: str = "res.png",
    save_dir: str = "outputs",
) -> None:
    "Draw bbox predictions on given image"

    # Extract the predicitons for given Image
    print("[INFO] Generating Predictions ..... ")
    bb, cls, sc = get_preds(model, img, score_thres, iou_thres)
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


@torch.no_grad()
def get_predictions_v2(model: nn.Module, image: np.array, thres: float) -> Tuple[List]:

    "get predictions for the uploaded image"
    # Convert Image to a tensor
    tensor_image = transforms(image=image)["image"]

    # Generate predicitons
    model.eval()
    pred = model([tensor_image])
    # Gather the bbox, scores & labels from the preds
    box = pred[0]["boxes"]  # Bounding boxes
    clas = pred[0]["labels"]  # predicted class labels
    score = pred[0]["scores"]  # predicted scores

    # Fit predicitons with score threshold
    detect_mask = score > thres
    box = box[detect_mask]
    clas = clas[detect_mask]
    score = score[detect_mask]

    box = list(box.cpu().numpy())
    clas = list(clas.cpu().numpy())
    score = list(score.cpu().numpy())
    return box, clas, score

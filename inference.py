import argparse
from typing import *

from references.data_utils import _get_logger
from references.utils import detection_api, get_model, load_yaml_config


def main(args):
    import warnings

    logger = _get_logger(__name__)
    warnings.filterwarnings("ignore")
    logger.info(f"Arguments: \n{args}")

    conf_dict = load_yaml_config(args.config)
    conf_dict["score_thres"] = args.score_thres
    conf_dict["nms_thres"] = args.iou_thres
    conf_dict["max_detections"] = args.md

    logger.info("Serializing model ")
    conf_dict = argparse.Namespace(**conf_dict)
    model = get_model(conf_dict)
    # grab the path to the Image file
    fname = args.image
    # get predictions
    detection_api(
        model,
        fname,
        save=args.save,
        show=args.show,
        fname=args.fname,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config/resnet34.yaml",
        required=False,
        help="path to the config file",
    )

    parser.add_argument(
        "--image", type=str, required=True, help="path to the input image"
    )

    parser.add_argument(
        "--score_thres",
        required=False,
        type=float,
        default=0.6,
        help="score_threshold to threshold detections",
    )

    parser.add_argument(
        "--iou_thres",
        required=False,
        type=float,
        default=0.5,
        help="iou_threshold for bounding boxes",
    )

    parser.add_argument(
        "--md",
        required=False,
        type=int,
        default=100,
        help="max detections in the image",
    )

    parser.add_argument(
        "--show",
        help="wether to display the output predicitons",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save",
        help="wether to save the ouput predictions",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        help="directory where to save the output predictions",
        default="output",
    )

    parser.add_argument(
        "--fname",
        type=str,
        required=False,
        help="name of the output prediction file",
        default="res.png",
    )

    args = parser.parse_args()
    main(args)

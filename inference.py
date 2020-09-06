import argparse
import warnings

import yaml

from utils import detection_api, get_model


def load_yaml_config(path):
    "load a yaml config"
    with open(path, "r") as f:
        conf_dict = yaml.safe_load(f)
    conf_dict = argparse.Namespace(**conf_dict)
    return conf_dict


def main(args):
    warnings.filterwarnings("ignore")
    # grab the model
    print("[INFO] Serializing model ....")
    conf_dict = load_yaml_config(args.config)
    model = get_model(conf_dict)

    # grab the path to the Image file
    fname = args.image

    # get predictions
    detection_api(
        model,
        fname,
        score_thres=args.score_thres,
        iou_thres=args.iou_thres,
        save=args.save,
        show=args.show,
        save_dir=args.save_dir,
        fname=args.fname,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
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
        default=0.5,
        help="score_threshold to threshold detections",
    )
    parser.add_argument(
        "--iou_thres",
        required=False,
        type=float,
        default=0.3,
        help="iou_threshold for bounding boxes",
    )
    parser.add_argument(
        "--save",
        type=bool,
        required=False,
        help="wether to save the ouput predictions",
        default=True,
    )
    parser.add_argument(
        "--show",
        type=bool,
        required=False,
        help="wether to display the output predicitons",
        default=False,
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

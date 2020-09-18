import ast
import glob
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import *

import pandas as pd

logger = logging.getLogger(__name__)


def xml_to_df(annot_path: str) -> pd.DataFrame:
    """
    Convert xml files to a pandas dataframe
    Args:
     annot_path: directory where the annotations are stored.
     img_dir   : directory where the images are stored.
    """
    xml_list = []
    for xml_file in glob.glob(annot_path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            value = (
                root.find("filename").text,
                int(root.find("size")[0].text),
                int(root.find("size")[1].text),
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text),
            )
            xml_list.append(value)
    column_name = ["filename", "width", "height", "xmin", "ymin", "xmax", "ymax"]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def read_dict(fname: str):
    """
    Reads in a dictionary from a file
    """
    f = open(fname, "r")
    label_dict = f.read()
    label_dict = ast.literal_eval(label_dict)
    f.close()
    return label_dict


def rev_dict(d: Dict) -> Dict:
    """
    Reverse a dictionary mapping
    """
    inv_dict = {v: k for k, v in d.items()}
    return inv_dict


def parse_data(img_dir: str, annot_dir: str, dict_path: str) -> pd.DataFrame:
    """
    Loads in the annotations from the xml files
    and generates a DataFrame. Where each entry corresponds
    to the path where images are stored and each image 
    has a bounding box denoted by xmin, ymin, xmax, ymax
    Integer targets are stored under targets.
    
    Args:
     img_dit   (str): Directory where the images are stored.
     annot_dir (str): Directory where the xml files are strored.
     dict_path (str): Path to the labels dictionary.
    
    """
    logger.info(f"Image Directory: {img_dir}")
    logger.info(f"Annotation Directory: {annot_dir}")
    logger.info(f"Path to labels: {dict_path}")

    # Convert the xml to dataframe
    df = xml_to_df(annot_dir)
    df["filename"] = [os.path.join(img_dir, f) for f in df["filename"].values]

    # Regular expression to grab the Class_name from the File_names
    pat = r"/([^/]+)_\d+.jpg$"
    pat = re.compile(pat)
    # Extract the Class_names
    df["classes"] = [pat.search(fname).group(1).lower() for fname in df.filename]

    # Read in the label dictionary
    label_dict = rev_dict(read_dict(fname=dict_path))
    # convert class labels to integers
    df["targets"] = [label_dict[idx] for idx in df["classes"].values]
    logger.info(f"Number of unique classes found: {len(df['targets'].unique())}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "img_dir", type=str, required=True, help="path to the image directory"
    )
    parser.add_argument(
        "annot_dir", type=str, required=True, help="path to the annotation directory"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        required=False,
        default="pet_labels.csv",
        help="path to the output csv file",
    )
    parser.add_argument(
        "labels",
        type=str,
        required=False,
        default="labels.names",
        help="path to the label dictionary",
    )
    args = parser.parse_args()

    df = parse_data(args.img_dir, args.annot_dir, args.labels)
    df.to_csv(args.output_dir, index=False)
    logger.info(f"Csv file save to {args.output_dir}")

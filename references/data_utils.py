import argparse
import ast
import glob
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import *

import pandas as pd
from sklearn.model_selection import train_test_split
from termcolor import colored


# Fancy Logging
def _get_logger(name=None):
    # Set up Logging
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logging.basicConfig(
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = _ColorfulFormatter(
        colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
        datefmt="%m/%d %H:%M:%S",
        root_name="retinanet_pet_detector",
        abbrev_name=str("rpd"),
    )

    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


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


def parse_data(img_dir: str, annot_dir: str, dict_path: str, logger=None) -> pd.DataFrame:
    """
    Loads in the annotations from the xml files
    and generates a DataFrame. Where each entry corresponds
    to the path where images are stored and each image 
    has a bounding box denoted by xmin, ymin, xmax, ymax
    Integer labels are stored under labels.
    
    Args:
     img_dit   (str): Directory where the images are stored.
     annot_dir (str): Directory where the xml files are strored.
     dict_path (str): Path to the labels dictionary.
    
    """
    if logger is None:
        logger = _get_logger(__name__)
        
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
    df["labels"] = [label_dict[idx] for idx in df["classes"].values]
    logger.info(f"Number of unique classes found: {len(df['labels'].unique())}")

    return df


def create_splits(df: pd.DataFrame, split_sz: float = 0.3, seed: int = 42):
    "Split given DataFrame into `split_sz`"
    # Grab the Unique Image Idxs from the Filename
    unique_ids = list(df.filename.unique())
    # Split the Unique Image Idxs into Train & valid Datasets
    try:
        train_ids, val_ids = train_test_split(
            unique_ids,
            shuffle=True,
            random_state=seed,
            test_size=split_sz,
            stratify=df.labels,
        )
    except:
        train_ids, val_ids = train_test_split(
            unique_ids, shuffle=True, random_state=seed, test_size=split_sz
        )
    # Create Splits on the DataFrame
    df["split"] = 0

    for i, idx in enumerate(df.filename.values):
        if idx in set(train_ids):
            df["split"][i] = "train"
        elif idx in set(val_ids):
            df["split"][i] = "val"

    # Split the DataFrame into Train and Valid DataFrames
    df_trn, df_val = df.loc[df["split"] == "train"], df.loc[df["split"] == "val"]

    # reset the index of the DataFrames
    df_trn, df_val = df_trn.reset_index(drop=True), df_val.reset_index(drop=True)

    # drop the extra redundent column
    df_trn.drop(columns=["split"], inplace=True)
    df_val.drop(columns=["split"], inplace=True)

    return df_trn, df_val

import argparse
import ast
import glob
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import *

import pandas as pd
from sklearn.model_selection import train_test_split

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


def create_splits(df: pd.DataFrame, split_sz: float = 0.3):
    "Split given DataFrame into `split_sz`"
    # Grab the Unique Image Idxs from the Filename
    unique_ids = list(df.filename.unique())
    # Split the Unique Image Idxs into Train & valid Datasets
    train_ids, val_ids = train_test_split(
        unique_ids, shuffle=True, random_state=42, test_size=split_sz
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

    df_trn, df_val = df_trn.reset_index(drop=True), df_val.reset_index(drop=True)
    # drop the extra redundent column
    df_trn.drop(columns=["split"], inplace=True)
    df_val.drop(columns=["split"], inplace=True)

    return df_trn, df_val


if __name__ == "__main__":
    import argparse
    import logging
    import os
    from sys import argv
    import warnings

    warnings.filterwarnings("ignore")

    # Set up Logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    action_choices = ["create", "split"]
    parser.add_argument("--action", choices=action_choices, default=action_choices[0])

    parser.add_argument(
        "--img_dir",
        type=str,
        required=(action_choices[0] in argv),
        help="path to the image directory",
    )
    parser.add_argument(
        "--annot_dir",
        type=str,
        required=(action_choices[0] in argv),
        help="path to the annotation directory",
    )

    parser.add_argument(
        "--labels",
        type=str,
        required=False,
        default="../labels.names",
        help="path to the label dictionary",
    )
    parser.add_argument(
        "--csv",
        default="../data/data-full.csv",
        type=str,
        required=(action_choices[1] in argv),
        help="path to the csv file",
    )
    parser.add_argument(
        "--valid_size",
        type=float,
        required=(action_choices[1] in argv),
        help="size of the validation set relative to the train set",
        default=0.2,
    )
    parser.add_argument(
        "--test_size",
        type=float,
        required=(action_choices[1] in argv),
        help="size of the test set relative to the validation set",
        default=0.5,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="../data/",
        help="path to the output csv file",
    )

    args = parser.parse_args()
    if action_choices[0] in argv:
        logger.info("Convert xml files to a csv file")
        df = parse_data(args.img_dir, args.annot_dir, args.labels)
        os.makedirs(args.output_dir, exist_ok=True)
        pth = os.path.join(args.output_dir, "data-full.csv")
        df.to_csv(pth, index=False)
        logger.info(f"csv file saved as {pth}")

    elif action_choices[1] in argv:
        logger.info(f"path to the given csv file : {args.csv}")
        logger.info("Creating train, validation and test splits")

        df = pd.read_csv(args.csv)
        # Create Splits in the DataFrame
        df_train, df_validation = create_splits(df, split_sz=args.valid_size)
        df_test, df_validation = create_splits(df, split_sz=args.test_size)

        l_tr, l_val, l_test = (
            len(df_train.filename.unique()),
            len(df_validation.filename.unique()),
            len(df_test.filename.unique()),
        )
        logger.info(f"Number of training examples={l_tr}")
        logger.info(f"Number of validation examples={l_val}")
        logger.info(f"Number of test examples={l_test}")

        df_train.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
        df_validation.to_csv(os.path.join(args.output_dir, "valid.csv"), index=False)
        df_test.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

        logger.info(f"Files saved to {args.output_dir}")

from references.data_utils import _get_logger

logger = _get_logger(name=__name__)


if __name__ == "__main__":
    import argparse
    import os
    from sys import argv
    import warnings
    from references.data_utils import create_splits, parse_data
    import pandas as pd

    warnings.filterwarnings("ignore")

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
        required=(action_choices[0] in argv),
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
        required=False,
        default=None,
        help="path to the output csv file",
    )
    parser.add_argument(
        "--seed", required=False, default=42, help="random seed", type=int,
    )

    args = parser.parse_args()

    if action_choices[0] in argv:
        logger.info("Converting xml files to a csv file")
        df = parse_data(args.img_dir, args.annot_dir, args.labels)
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            pth = os.path.join(args.output_dir, "data-full.csv")
            df.to_csv(pth, index=False)
            logger.info(f"csv file saved as {pth}")
        else:
            df.to_csv("data-full.csv", index=False)
            logger.info("csv file saved to current directory as `data-full.csv`")

    elif action_choices[1] in argv:
        logger.info(f"Path to the given csv file : {args.csv}")
        logger.info("Generatind train, validation and test splits")

        df = pd.read_csv(args.csv)
        # Create Splits in the DataFrame
        # create training & validation splits from the train and validation dataset
        df_train, df_validation = create_splits(df, split_sz=args.valid_size, seed=args.seed)
        # create test and validation splits from the validation and test dataset
        df_test, df_validation = create_splits(df_validation, split_sz=args.test_size, seed=args.seed)

        if args.output_dir is not None:
            df_train.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
            df_validation.to_csv(os.path.join(args.output_dir, "valid.csv"), index=False)
            df_test.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)
            logger.info(f"Files saved to {args.output_dir}")
        else:
            df_train.to_csv("train.csv", index=False)
            df_validation.to_csv("valid.csv", index=False)
            df_test.to_csv("test.csv", index=False)
            logger.info("Files saved to current directory")

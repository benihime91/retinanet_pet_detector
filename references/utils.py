import importlib
from typing import *

import albumentations as A
from omegaconf import DictConfig, OmegaConf


def _load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
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


def _get_tfms(conf: DictConfig) -> Dict[str, A.Compose]:
    """
    Loads in albumentation augmentations for train, valid, test
    from given config as a dictionary
    """
    trn_tfms = [
        _load_obj(i["class_name"])(**i["params"]) for i in conf.augmentation.train
    ]
    val_tfms = [
        _load_obj(i["class_name"])(**i["params"]) for i in conf.augmentation.valid
    ]
    test_tfms = [
        _load_obj(i["class_name"])(**i["params"]) for i in conf.augmentation.test
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

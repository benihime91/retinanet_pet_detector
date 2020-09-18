config_path = "/Users/ayushman/Desktop/retinanet_pet_detector/config/main.yaml"

from omegaconf import DictConfig, OmegaConf

config = OmegaConf.load(config_path)
# print(config.pretty())
print(config.hparams.pretty())


cfg = {
    "general": {"save_dir": "logs/", "workspace": "erlemar", "project_name": "wheat"},
    "dataset": {"class_name": "WheatDataset"},
    "trainer": {
        "gpus": 1,
        "distributed_backend": "dp",
        "accumulate_grad_batches": 1,
        "profiler": False,
        "max_epochs": 13,
        "gradient_clip_val": 0.5,
        "num_sanity_val_steps": 0,
        "weights_summary": None,
    },
    "training": {
        "lr": 0.0001,
        "metric": "main_score",
        "seed": 666,
        "debug": False,
        "mode": "max",
    },
    "logging": {"log": True},
    "optimizer": {
        "class_name": "torch.optim.AdamW",
        "params": {"lr": "${training.lr}", "weight_decay": 0.001},
    },
    "scheduler": {
        "class_name": "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "step": "epoch",
        "monitor": "${training.metric}",
        "params": {"mode": "${training.mode}", "factor": 0.1, "patience": 5},
    },
    "model": {
        "backbone": {
            "class_name": "torchvision.models.detection.fasterrcnn_resnet50_fpn",
            "params": {"pretrained": True},
        },
        "head": {
            "class_name": "torchvision.models.detection.faster_rcnn.FastRCNNPredictor",
            "params": {"num_classes": 2},
        },
    },
    "callbacks": {
        "early_stopping": {
            "class_name": "pl.callbacks.EarlyStopping",
            "params": {
                "monitor": "${training.metric}",
                "patience": 10,
                "mode": "${training.mode}",
            },
        },
        "model_checkpoint": {
            "class_name": "pl.callbacks.ModelCheckpoint",
            "params": {
                "monitor": "${training.metric}",
                "save_top_k": 3,
                "filepath": "saved_models/",
                "mode": "${training.mode}",
            },
        },
    },
    "private": {"comet_api": "fOmVZaafsPuJ6OP3myaJUd4fC"},
    "data": {
        "folder_path": "/kaggle/input/global-wheat-detection",
        "num_workers": 0,
        "batch_size": 12,
    },
    "augmentation": {
        "train": {
            "augs": [
                {"class_name": "albumentations.Flip", "params": {"p": 0.6}},
                {
                    "class_name": "albumentations.RandomBrightnessContrast",
                    "params": {"p": 0.6},
                },
                {
                    "class_name": "albumentations.pytorch.transforms.ToTensorV2",
                    "params": {"p": 1.0},
                },
            ],
            "bbox_params": {"format": "pascal_voc", "label_fields": ["labels"]},
        },
        "valid": {
            "augs": [
                {
                    "class_name": "albumentations.pytorch.transforms.ToTensorV2",
                    "params": {"p": 1.0},
                }
            ],
            "bbox_params": {"format": "pascal_voc", "label_fields": ["labels"]},
        },
    },
}


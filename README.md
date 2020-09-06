# retinanet_pet_detector

Using a `Retinanet` to detect `cats & dogs`.

Create a `PetDetector` which can detect the `faces` of cats & dogs in Images using my implementation of [Retinanet](https://github.com/benihime91/pytorch_retinanet).

## **Dataset used**:

`The Oxford-IIIT Pet Dataset` which can be found here [dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## **TODO**:

- [x] Parse the data and convert it to a managable format ex: CSV.
- [x] Finish [Retinanet Project](https://github.com/benihime91/pytorch_retinanet) first.
- [x] Train the Network.
- [x] Create WebApp using `StreamLit`.

## **Usage**:

- Install [python3](https://www.python.org/downloads/)

- Install dependencies

  ```bash
  git clone https://github.com/benihime91/retinanet_pet_detector.git
  cd retinanet_pet_detector
  pip install -r requirements.txt
  ```

- Run app

  ```bash
  streamlit run app.py
  ```

## **Train**:

To train from scratch check the nbs folder for the notebooks. There are two notebooks to using:

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/benihime91/1a413db4b8ac33918d949ec5f57b51b3/03_train.ipynb?authuser=4) [03_train.ipynb](https://github.com/benihime91/retinanet_pet_detector/blob/master/nbs/03_train.ipynb). Train using same hyperparameters with `resnet18` backbone.

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/benihime91/02481b7f5c338fe835f4af6e9316d47e/04_template.ipynb?authuser=4) [04_template.ipynb](https://github.com/benihime91/retinanet_pet_detector/blob/master/nbs/04_template.ipynb). Template notebook to train using different hyperparameters and different backbone.

## **Inference on Single Image**:

A pretrained model with `resnet18` backbone is available at: [weights](https://github.com/benihime91/retinanet_pet_detector/releases/download/v0.0.1/resnet18-2020-08-04-ffdde352.pth).<br>

Using inference.py automatically loads in these weights on a retinanet with `resnet18` backbone.

```bash
  python inference.py --image "/Users/ayushman/Desktop/Datasets/oxford-iiit-pet/images/great_pyrenees_19.jpg" --fname res_test.png
```

- To change the model backbone , model loading path/url modify the `config.yaml` file.

- Results are automatically saved to `output/{--fname}` to change this modify the flags of `inference.py`

  **Flags**:

  ```bash
  > python inference.py --help
  usage: inference.py [-h] [--config CONFIG] --image IMAGE
                    [--score_thres SCORE_THRES] [--iou_thres IOU_THRES]
                    [--save SAVE] [--show SHOW] [--save_dir SAVE_DIR]
                    [--fname FNAME]

  optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       path to the config file
  --image IMAGE         path to the input image
  --score_thres SCORE_THRES
                        score_threshold to threshold detections
  --iou_thres IOU_THRES
                        iou_threshold for bounding boxes
  --save SAVE           wether to save the ouput predictions
  --show SHOW           wether to display the output predicitons
  --save_dir SAVE_DIR   directory where to save the output predictions
  --fname FNAME         name of the output prediction file
  ```

## **Exploratory Data Analysis**:

![cat_breeds](nbs/Ims/cat_breeds.png) ![cat_breeds](nbs/Ims/dog_breeds.png)

> **Example images from the Dataset:**

![example_1](nbs/Ims/example.png) ![example_1](nbs/Ims/example_2.png)

## **Results**:

- **COCO API results on hold-out test dataset with a `resnet18` backbone:**

  ```bash
  IoU metric: bbox
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.434
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.938
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.312
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.500
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.413
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.406
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.463
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.463
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.500
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.450
  ```

  **Results can be better with a bigger backbone.**

- **Results:** ![](pets_logs/res_2.png) ![](pets_logs/res_3.png) ![](pets_logs/res_4.png)

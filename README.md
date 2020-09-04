# retinanet_pet_detector
Using a `Retinanet` to detect `cats & dogs`.

Create a `PetDetector` which can detect the `faces` of cats & dogs in Images using my implementation of [Retinanet](https://github.com/benihime91/pytorch_retinanet).

## Dataset used :
`The Oxford-IIIT Pet Dataset` which can be found here [dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## TODO: 
- [x] Parse the data and convert it to a managable format ex: CSV.
- [x] Finsh [Retinanet Project](https://github.com/benihime91/pytorch_retinanet) first.
- [x] Train the Network.
- [ ] Create WebApp using `StreamLit`.

## Exploratory Data Analysis:
![cat_breeds](nbs/Ims/cat_breeds.png)
![cat_breeds](nbs/Ims/dog_breeds.png)  

>**Example images from the Dataset:**  

![example_1](nbs/Ims/example.png)
![example_1](nbs/Ims/example_2.png)

## Results:
- **COCO API results on hold-out test dataset:**
```bash
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.594
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.919
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.584
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.800
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.586
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.612
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.654
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.654
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.800
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.642
```
- **Training Logs:**

  <img src="pets_logs/pets_total_loss.png" width="400" height="400" />
  <img src="pets_logs/pets_classification_loss.png" width="400" height="400" />
  <img src="pets_logs/pets_regression_loss.png" width="400" height="400" />
  <img src="pets_logs/pets_valid_mAP.png" width="400" height="400" />
  
- **Results:**
 
  <img src="pets_logs/res_2.png" width="500" height="500" />
  <img src="pets_logs/res_3.png" width="500" height="500" />
  <img src="pets_logs/res_4.png" width="500" height="500" />

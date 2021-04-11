# Week4 - Object Detection and Segmentation

The goal of this week is to use the Detectron2 framework by running inference and training with different Mask R-CNN models on the KITTI-MOTS AND MOTSChallege datasets, and analyze the object detection and segmentation results. All the models are pre-trained with the COCO dataset except one, that is trained both COCO and Cityscapes dataset.

## Tasks

The main tasks are divided in three files: task_a, task_b and task_c, but other files are included to help obtaining the results provided in the slides.

* Task a: Running inference with the pretrained models.
* Task b: Pretrain the best model from task a and analyze the results.
* Task c: Fine tune the best model from task a with cross-validation, comparing the effect that each hyperparameter has on the results.

Running each main task is as simple as

````
python task_#.py
````

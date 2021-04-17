# Week5 - Challenges of Object Detection and Segmentation

The goal of this week is to use the Detectron2 framework by running inference on images with Out of Context objects to analyze the object detection and segmentation results.

## Tasks

* Task (a) Running inference on [Out of Context](http://people.csail.mit.edu/myungjin/publications/outOfContext.pdf) dataset.

Based on the [The Elephant in the Room](https://arxiv.org/pdf/1808.03305.pdf) paper:
* Task (b) Transplant new objects from COCO dataset classes to COCO test dataset images to test co-occurrency.
* Task (c) Qualitative transplant: duplicate objects from within the COCO test dataset images and copy them to another locations in the same images.
* Task (d) Apply feature interference.

Based on [Imagenet-trained CNNs are biased towards texture](https://arxiv.org/pdf/1811.12231v2.pdf):
* Task (e) Apply [Image Style Transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

All the tasks use the same code to run inference on the modified datasets:

 ```bash
 python inference_qualitative.py 
 
 usage: inference_qualitative.py.py [--model] [--data] [--output]

arguments:
  --model               model used: Faster or Mask R-CNN
  --data                data path: './Data/task_#'
  --output              outputs results path

```


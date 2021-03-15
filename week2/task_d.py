# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, sys
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from LossEvalHook import *
from MyTrainer import *
#from PlotTogether import *

### ToDo:  See how to handle the split in train, val and test. There is a txt file listing the images for each one
#split_path = '/home/mcv/datasets/KITTI/'
#dataset_path = '/home/mcv/datasets/KITTI/data_object_image_2/training/image_2/'
#gt_path= '/home/mcv/datasets/KITTI/training/label_2'
def kitti_dataset(dataset_type):
    '''
    #Values    Name      Description
    ----------------------------------------------------------------------------
       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc' or 'DontCare'
       1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
       1    score        Only for results: Float, indicating confidence in
                         detection, needed for p/r curves, higher is better.
    '''
    classes = {
        'Car' : 0,
        'Van' : 1,
        'Truck' : 2,
        'Pedestrian' : 3,
        'Person_sitting' : 4,
        'Cyclist' : 5,
        'Tram' : 6,
        'Misc' : 7,
        'DontCare' : 8
    }

    kitti_dicts = []
    split_path = '/home/mcv/datasets/KITTI/'
    dataset_path = '/home/mcv/datasets/KITTI/data_object_image_2/training/image_2/'
    gt_path= '/home/mcv/datasets/KITTI/training/label_2/'

    split_file = split_path+'{}_kitti.txt'.format(dataset_type)
    dataset_list = []
    with open(split_file) as split_f:
        for line in split_f:
            dataset_list.append(line.strip())

    # my guess here read the split_path/[train,val,test] txt file and put it on a list
    for img_id, img_name in enumerate(dataset_list):#change dataset_path for read split list of files
        record = {}

        filename = dataset_path + img_name.replace(".txt", ".png")
        height, width = cv2.imread(filename).shape[:2]

        record['file_name'] = filename
        record['image_id'] = img_id
        record['height'] = height
        record['width'] = width

        gt_file = gt_path + img_name
        objs = []
        with open(gt_file) as gt_f:
            for line in gt_f:
                gt=line.strip().split(' ')

                gt_type = gt[0]
                bbox_left = float(gt[4])
                bbox_top = float(gt[5])
                bbox_right = float(gt[6])
                bbox_bottom = float(gt[7])

                obj = {
                    "bbox" : [bbox_left, bbox_top, bbox_right, bbox_bottom],
                    "bbox_mode" : BoxMode.XYXY_ABS,
                    "category_id" : classes[gt_type]
                }
                objs.append(obj)
        record['annotations'] = objs
        kitti_dicts.append(record)

    return kitti_dicts


if __name__ == "__main__":

    model_path = '../models/'
    results_path = '../results/task_d/to_plot/'

    """
    Register the custom dataset to detectron2, following the detectron2 custom dataset tutorial.
    Here, the dataset is in its custom format, therefore we write a function to parse it and
    prepare it into detectron2's standard format. User should write such a function when using
    a dataset in custom format. See the tutorial for more details.
    """
    # split training into three parts. training, validation and testing
    for d in ['train', 'val','test']:#'test',
        DatasetCatalog.register("kitti_{}".format(d), lambda d = d : kitti_dataset(d))
        MetadataCatalog.get("kitti_{}".format(d)).set(
            thing_classes=['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
                           'Cyclist', 'Tram', 'Misc', 'DontCare'])
    kitti_metadata = MetadataCatalog.get("kitti_train")

    ###-------TRAIN-----------------------------

    # SELECT MODEL: If enough time, test different models and compare results and execution time
    # Best accuracy: model = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    # Fastest:  model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" #BASE_LR=

    model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = ("kitti_train",)
    cfg.DATASETS.VAL = ('kitti_val',)
    cfg.DATASETS.TEST = ('kitti_test',)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.OUTPUT_DIR = results_path + model.split('COCO-Detection/')[1].split('.')[0] + "/lr_0001/"
    cfg.SOLVER.MAX_ITER = 1000

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for the tutorial dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  # number of classes
    cfg.TEST.EVAL_PERIOD = 20 #eval_period

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(cfg.OUTPUT_DIR)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    ###-------INFERENCE AND EVALUATION---------------------------

    # First, let's create a predictor using the model we just trained:
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg.DATASETS.TEST = ("kitti_val",)
    predictor = DefaultPredictor(cfg)

    ### VISUALIZATION
    dataset_dicts = kitti_dataset("val")
    for i, d in enumerate(random.sample(dataset_dicts, 3)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(
            im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1], metadata=kitti_metadata,
                       scale=0.5)  # instance_mode=ColorMode.IMAGE_BW remove the colors of unsegmented pixels.         #This option is only available for segmentation models
        out_test_img = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        cv2.imwrite(cfg.OUTPUT_DIR + '/visualization/kitti_val_sample_{}.png'.format(i), out_test_img.get_image()[:, :, ::-1])

    ### MAP #####
    #We can also evaluate its performance using AP metric implemented in COCO API.
    evaluator = COCOEvaluator("kitti_test", cfg, False, output_dir=results_path)
    val_loader = build_detection_test_loader(cfg, "kitti_test")
    print('---------------------------------------------------------')
    print(model)
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
    print('---------------------------------------------------------')

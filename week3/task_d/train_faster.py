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

import argparse

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

sys.path.append('/home/group02/week3/code')
from mots_utils import *
from LossEvalHook import *
from MyTrainer import *

def read_annotations(gt):
    patterns = list(np.unique(gt))[1:-1]

    objs = []
    for pattern in patterns:
        coords = np.argwhere(gt==pattern)

        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0)

        bbox = [y0, x0, y1, x1]

        obj = {
            "bbox": bbox,
            "bbox_mode":BoxMode.XYXY_ABS,
            "category_id": map_classes[int(np.floor(gt[coords[0][0]][coords[0][1]]/1e3))],
            "iscrowd": 0
        }

        objs.append(obj)

    return objs

def load_dataset(type, set_config, thing_classes, map_classes):
    dataset_path = {
        'KITTI-MOTS' : '/home/group02/mcv/datasets/KITTI-MOTS/training/image_02',
        'MOTSChallenge' : '/home/group02/mcv/datasets/MOTSChallenge/train/images'
    }
    gt_path = {
        'KITTI-MOTS' : '/home/group02/mcv/datasets/KITTI-MOTS/instances',
        'MOTSChallenge' : '/home/group02/mcv/datasets/MOTSChallenge/train/instances'
    }

    if type == 'test':
        filepath = '/home/group02/week3/data/split/kitti_mots_test.txt'
    else:
        filepath = '/home/group02/week3/data/cros_val/' + type + '_' + set_config + '.txt'


    with open(filepath, 'r') as f:
        lines = [line.rstrip() for line in f]

    dataset_seqs = []
    for l in lines:
        dataset = l.split('datasets/')[1].split('/')[0]
        seq = l.split('/')[-2]
        dataset_seqs.append([dataset, seq])

    dataset_dicts = []
    for dataset, seq in dataset_seqs:
        for j, img_name in enumerate(os.listdir(os.path.join(dataset_path[dataset], seq))):

            if 'png' not in img_name and 'jpg' not in img_name:
                continue

            record = {}
            filename = os.path.join(dataset_path[dataset], seq, img_name)
            gt_filename = os.path.join(gt_path[dataset], seq, img_name.split('.')[0]+'.png')

            gt = np.asarray(Image.open(gt_filename))

            height, width = gt.shape[:]

            record["file_name"] = filename
            # record["image_id"] = i+j
            record["image_id"] = filename
            record["height"] = height
            record["width"] = width

            record["annotations"] = read_annotations(gt)
            dataset_dicts.append(record)

    return dataset_dicts

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='faster_rcnn_R_101_FPN_3x',
                        help='pre-trained model to run inference on KITTI-MOTS dataset')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')

    parser.add_argument('--iter', type=int, default=5000,
                        help='max iterations (epochs)')

    parser.add_argument('--batch', type=int, default=512,
                        help='batch size')

    parser.add_argument('--set_config', type=str, default='0',
                        help='which configuration of cross validation to use')

    return parser.parse_args(args)

if __name__ == "__main__":

    args = parse_args()

    model = 'COCO-Detection/' + args.model + '.yaml'
    print('[INFO] Using model: ', model)

    ###-------TRAIN-----------------------------
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    cfg.OUTPUT_DIR = '/home/group02/week3/results/' + args.model + '/lr_' + str(args.lr).replace('.', '_') + '_iter_' + str(args.iter) + '_batch_' + str(args.batch) + '/' + args.set_config + '/'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    thing_classes = ['Car', 'Pedestrian']
    map_classes = {1:0, 2:1}
    dataset='KITTI-MOTS'

    for d in ['train', 'val', 'test']:
        DatasetCatalog.register(dataset + '_' + d, lambda d=d: load_dataset(d, args.set_config, thing_classes, map_classes))
        MetadataCatalog.get(dataset + '_' + d).set(thing_classes=thing_classes)

    metadata = MetadataCatalog.get(dataset + '_train')

    cfg.DATASETS.TRAIN = (dataset + '_train',)
    cfg.DATASETS.VAL = (dataset + '_val',)
    cfg.DATASETS.TEST = (dataset + '_test',)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.SOLVER.IMS_PER_BATCH = 2

    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch   # faster, and good enough for the tutorial dataset (default: 512)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    ###-------INFERENCE AND EVALUATION---------------------------
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model

    ### MAP #####
    #We can also evaluate its performance using AP metric implemented in COCO API.
    evaluator = COCOEvaluator(dataset + '_val', cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, dataset + '_val')
    print('---------------------------------------------------------')
    print(model)
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
    print('---------------------------------------------------------')

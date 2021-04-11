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


def read_annotations(gt):
    patterns = list(np.unique(gt))[1:-1]

    objs = []
    for pattern in patterns:
        coords = np.argwhere(gt==pattern)

        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0)

        bbox = [y0, x0, y1, x1]

        copy = gt.copy()
        copy[gt==pattern] = 255
        copy[gt!=pattern] = 0
        copy = np.asarray(copy,np.uint8)

        contours, _ = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        contour = [np.reshape(contour,(contour.shape[0],2)) for contour in contours]
        contour = np.asarray([item for tree in contour for item in tree])
        px = contour[:,0]
        py = contour[:,1]
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x]

        if len(poly) < 6:
            continue


        obj = {
            "bbox": bbox,
            "bbox_mode":BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": map_classes[int(np.floor(gt[coords[0][0]][coords[0][1]]/1e3))],
            "iscrowd": 0
        }

        objs.append(obj)

    return objs


def load_dataset(type, thing_classes, map_classes):
    dataset_path = {
        'KITTI-MOTS' : '/home/group02/mcv/datasets/KITTI-MOTS/training/image_02',
        'MOTSChallenge' : '/home/group02/mcv/datasets/MOTSChallenge/train/images'
    }
    gt_path = {
        'KITTI-MOTS' : '/home/group02/mcv/datasets/KITTI-MOTS/instances',
        'MOTSChallenge' : '/home/group02/mcv/datasets/MOTSChallenge/train/instances'
    }

    with open('/home/group02/week3/data/split/kitti_mots_' + type + '.txt', 'r') as f:
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

    parser.add_argument('--config', type=str, default='COCO-InstanceSegmentation',
                        help='configuration of the pre-trained model')

    parser.add_argument('--model', type=str, default='mask_rcnn_X_101_32x8d_FPN_3x',
                        help='pre-trained model to run inference on KITTI-MOTS dataset')

    parser.add_argument('--output', type=str, default='/home/group02/week4/results/task_a',
                        help='output path to store the quantitative results')

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()

    model_path = os.path.join(args.config, args.model + '.yaml')
    print('[INFO] Using model: ', model_path)

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)

    cfg.OUTPUT_DIR = os.path.join(args.output, args.config, args.model)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    thing_classes = metadata.thing_classes
    map_classes = {1:2,2:0}
    dataset='KITTI-MOTS'

    for d in ['train', 'test']:
        DatasetCatalog.register(dataset + '_' + d, lambda d=d: load_dataset(d,thing_classes,map_classes))
        MetadataCatalog.get(dataset + '_' + d).set(thing_classes=thing_classes)

    metadata = MetadataCatalog.get(dataset + '_train')

    cfg.DATASETS.TRAIN = (dataset + '_train',)
    cfg.DATASETS.TEST = (dataset + '_test',)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    evaluator = COCOEvaluator(dataset + '_test', cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, dataset + '_test')
    print('---------------------------------------------------')
    print('Evaluation with model ', model_path)
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
    print('---------------------------------------------------')

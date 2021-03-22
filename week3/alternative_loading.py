import json, random, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

from mots_utils import load_txt, rletools, load_images_for_folder

# Only this path has to be changed
dataset_path = Path('/home/adityassrana/MCV_UAB/m5-vr/project/week3_dev/Datasets')

mots_path = dataset_path/'MOTSChallenge'
kitti_path = dataset_path/'KITTI-MOTS'

training_seqs_kitti_mots = [11,17,9,20,19,5,0,15,1,4,3,12]
training_seqs_mots_challenge = [2,11,5,9]
testing_seqs_kitti_mots = [2,6,7,8,10,13,14,16,18]

def get_training_files():
    training_image_paths = []
    training_instances_path = []

    for seq in training_seqs_kitti_mots:
        training_image_paths.append(str(dataset_path/kitti_path/'training/image_02'/str(seq).zfill(4)))
        training_instances_path.append(str(dataset_path/kitti_path/'instances_txt'/str(seq).zfill(4))+'.txt')

    for seq in training_seqs_mots_challenge:
        training_image_paths.append(str(dataset_path/mots_path/'train/images'/str(seq).zfill(4)))
        training_instances_path.append(str(dataset_path/mots_path/'train/instances_txt'/str(seq).zfill(4))+'.txt')

    training_image_folders = sorted(training_image_paths)
    training_instances_txts = sorted(training_instances_path)
    return [(folder,txt) for folder,txt in zip(training_image_folders, training_instances_txts)]

def get_testing_files():
    testing_image_paths = []
    testing_instances_path = []

    for seq in testing_seqs_kitti_mots:
        testing_image_paths.append(str(dataset_path/kitti_path/'training/image_02'/str(seq).zfill(4)))
        testing_instances_path.append(str(dataset_path/kitti_path/'instances_txt'/str(seq).zfill(4))+'.txt')

    testing_image_folders = sorted(testing_image_paths)
    testing_instances_txts = sorted(testing_instances_path)
    return [(folder,txt) for folder,txt in zip(testing_image_folders, testing_instances_txts)]

def get_dataset_dicts():
    dataset_dicts = []
    for train_folder, train_txt in get_training_files():
        # get data folder and its corresponding txt file
        # load the annotations for the folder
        annotations = load_txt(train_txt)
        image_paths = sorted(os.listdir(train_folder))
        for indx, (image_path, (file_id, objects)) in enumerate(zip(image_paths, list(annotations.items()))):
            #check the file is png or jpg
            if image_path.split('.')[1] in ['png','jpg']:
                record = {}

                filename = os.path.join(train_folder, image_path)
                height,width = cv2.imread(filename).shape[:2]

                record["file_name"] = filename
                record["image_id"] = filename
                record["height"] = height
                record["width"] = width

                objs = []
                for obj in objects:
                    if obj.track_id != 10000:
                        category_id = obj.class_id    
                        bbox = rletools.toBbox(obj.mask)

                        obj_dic = {
                            "bbox" : list(bbox),
                            "bbox_mode" : BoxMode.XYWH_ABS,
                            "category_id" : category_id
                        }
                        objs.append(obj_dic)

                record["annotations"] = objs
                dataset_dicts.append(record)
    return dataset_dicts

DatasetCatalog.register("kitti_train", get_dataset_dicts)

MetadataCatalog.get("kitti_train").set(things_classes="cars,pedestrains")
kitti_metadata = MetadataCatalog.get("kitti_train")
dataset_dicts = get_dataset_dicts()

for indx,d in enumerate(random.sample(dataset_dicts, 1)):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_metadata, scale=1.2)
    out = visualizer.draw_dataset_dict(d)
    plt.figure(figsize=(16,8))
    plt.imshow(out.get_image()[:, :, ::-1])
    #cv2.imwrite(f'test_{indx}.png',out.get_image()[:, :, ::-1])
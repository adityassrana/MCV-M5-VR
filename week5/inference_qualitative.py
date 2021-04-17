import sys, os, cv2
import torch
assert torch.__version__.startswith("1.7")
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import argparse

models = {
    'faster': 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
    'mask': 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
}

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='mask',
                        choices=['faster', 'mask'],
                        help='pre-trained model to run inference on out-of-context dataset')

    parser.add_argument('--data', type=str, default='./data/task_a',
                        help='data path')

    parser.add_argument('--output', type=str, default='./results/task_a',
                        help='output path to store the qualitative results')

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()

    model_path = models[args.model]
    print('[INFO] Using model: ', model_path)

    # Run a pre-trained detectron2 model
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
    predictor = DefaultPredictor(cfg)

    cfg.OUTPUT_DIR = os.path.join(args.output, args.model)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    for subdir, dirs, files in os.walk(args.data):
        for file in files:
            input_path = os.path.join(subdir, file)
            output_path = os.path.join(cfg.OUTPUT_DIR, file)

            im = cv2.imread(input_path)
            outputs = predictor(im)

            # We can use `Visualizer` to draw the predictions on the image.
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

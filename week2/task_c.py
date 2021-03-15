import os, cv2
import torch
assert torch.__version__.startswith("1.7")   # need to manually install torch 1.8 if Colab changes its default version
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes

# Run a pre-trained detectron2 model
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

results_dir = '../results/task_c/'
data_path = '/home/mcv/datasets/MIT_split/'

# only detections with a confidence higher than this threshold are considered
conf_threshold = 0.5

for subdir, dirs, files in os.walk(data_path):
    results_path = os.path.join(results_dir, subdir.split('datasets/')[1])
    os.makedirs(results_path, exist_ok=True)
    for file in files:

        filtered_boxes = []
        filtered_scores = []
        filtered_classes = []

        input_path = os.path.join(subdir, file)
        output_path = results_path+file

        im = cv2.imread(input_path)

        outputs = predictor(im)

        pred_boxes = outputs["instances"].pred_boxes.to("cpu")
        scores = outputs["instances"].scores.to("cpu")
        pred_classes = outputs["instances"].pred_classes.to("cpu")

        for idx, s in enumerate(scores):
            if s.item() >= conf_threshold:
                filtered_boxes.append(pred_boxes[idx].tensor.numpy()[0])
                filtered_scores.append(scores[idx].item())
                filtered_classes.append(pred_classes[idx].item())

        filtered_outputs = Instances([256,256], pred_boxes=torch.tensor(filtered_boxes), scores=torch.tensor(filtered_scores), pred_classes=torch.tensor(filtered_classes))

        # print(outputs["instances"].pred_classes)
        # print(outputs["instances"].pred_boxes)

        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(filtered_outputs)
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

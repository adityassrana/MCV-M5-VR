from mots_utils import *
from detectron2.structures import BoxMode

def read_annotations(annotation_path):
    annotation = load_txt(annotation_path)

    objs = {}
    for frame_id, objects in annotation.items():
        frame_objs = []
        for track in objects:
            if track.track_id != 10000:

                class_id = track.track_id //1000 # or track.class_id
                instance_id = track.track_id % 1000

                bbox = rletools.toBbox(track.mask)

                coco_bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])

                obj = {
                    "bbox" : coco_bbox,
                    "bbox_mode" : BoxMode.XYXY_ABS,
                    "category_id" : class_id
                }
                frame_objs.append(obj)
        if len(frame_objs) != 0:
            objs[frame_id] = frame_objs

    return objs

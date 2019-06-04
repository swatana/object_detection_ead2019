"""Mask RCNN"""

import cv2
import numpy as np
from PIL import Image

from mrcnn import model
from mrcnn import visualize
from mrcnn.config import Config

class InferenceConfig(Config):
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def _read_lines(filepath):
    with open(filepath) as fp:
        return [line.strip() for line in fp]


def _make_config(base_config, num_classes, threshold, iou_threshold):
    class CustomConfig(base_config):
        NUM_CLASSES = num_classes
        DETECTION_MIN_CONFIDENCE = threshold
        DETECTION_NMS_THRESHOLD = iou_threshold

    return CustomConfig()


class MaskRCNN(object):
    def __init__(self, model_path, classes_path, threshold=0.7, iou_threshold=0.3):
        self.model_path = model_path
        self.classes_path = classes_path
        self.class_names = _read_lines(classes_path)
        self.model = model.MaskRCNN(
            mode="inference",
            config=_make_config(InferenceConfig, len(self.class_names), threshold, iou_threshold),
            model_dir="",  # not necessary for inference mode
        )
        self.model.load_weights(model_path, by_name=True)

    def detect_image(self,
                     image,
                     font_path=None,
                     image_annotation=False,
                     extract_feature=False):
        """Detect objects in an image.

        This is made to resemble yolo3.

        Args:
        - font_path: ignored.
        - extract_feature: ignored.
        """

        # Convert PIL to np.ndarray
        image = np.array(image)

        output = self.model.detect([image])[0]  # 0.4 seconds/image
        boxes = output['rois']
        class_ids = output['class_ids']
        masks = output['masks']
        scores = output['scores']

        objects = []

        trspsd_masks = np.transpose(masks, (2, 0, 1))

        for box, score, cls, mask in zip(boxes, scores, class_ids, trspsd_masks):
            predicted_class = self.class_names[cls]
            top, left, bottom, right = box.astype(int)
            objects.append({
                "bbox": [left, top, right, bottom],
                "score": np.asscalar(score),
                "class_name": predicted_class,
                "class_id": cls,
                "mask": mask
            })

        result = {
            "objects": objects
        }

        return result

    def close_session(self):
        """Do nothing"""

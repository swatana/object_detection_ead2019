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

        # r_img, polygons = self.__draw_boxes(image, boxes, class_ids, masks, scores)

        result = {
            "objects": objects
        }

        return result

    def __make_json_annotation(self, boxes, polygons, class_ids, scores):
        result = []
        for [y1, x1, y2, x2], polygon, class_id, score in zip(boxes, polygons, class_ids, scores):
            bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            result.append({
                "bbox": np.array(bbox).tolist(),
                "polygon": polygon,
                "score": np.asscalar(score),
                "class": self.class_names[class_id],
            })
        return result

    def __draw_boxes(self, img, boxes, class_ids, masks, scores, alpha=0.3):
        """Draw bounding boxes.

        Args:
        - image: np.ndarray image in PIL format
        """

        # Convert to OpenCV format
        img = np.array(img[..., ::-1], dtype=np.float32)

        masks = np.transpose(masks, (2, 0, 1))
        obj_count = masks.shape[0]
        colors = [[int(a * 255) for a in color[::-1]]
                  for color in visualize.random_colors(obj_count)]
        polygons = []
        for color, mask, (y1, x1, y2, x2), class_id, score in zip(
                colors, masks, boxes, class_ids, scores):
            img[mask] = (img[mask] * (1.0 - alpha) + np.array(color) * alpha)

            # Draw mask boundaries
            contours = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[-2]

            polygon = []

            for cnt in contours:
                ps=[]
                if (len(cnt) <= 2):
                    continue
                for point in cnt:
                    x, y = point[0]
                    ps.append(int(x))
                    ps.append(int(y))
                polygon.append(ps)

            polygons.append(polygon)

            cv2.drawContours(img, contours, -1, color, 2)

            # Draw rectangular bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Label
            cv2.putText(img, "%i %s %.2f" % (class_id, self.class_names[class_id], score),
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 255, 255))

        img = img[..., ::-1]  # Convert back to PIL
        return Image.fromarray(img.astype(np.uint8)), polygons

    def close_session(self):
        """Do nothing"""

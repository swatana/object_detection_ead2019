import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2

from keras_centernet.models.networks.hourglass import HourglassNetwork, normalize_image
from keras_centernet.models.decode import CtDetDecode
from keras_centernet.utils.utils import coco_names
from keras_centernet.utils.letterbox import LetterboxTransformer

def _read_lines(filepath):
    with open(filepath) as fp:
        return [line.strip() for line in fp]

def pil2cv(image):
    new_image = np.asarray(image)[:, :, ::-1].copy()
    return new_image

class CENTERNET(object):
    def __init__(self, model, classes_path):
        self.weights = model
        self.classes_name = _read_lines(classes_path)
        if self.classes_name is None:
            self.classes_name = coco_names
        kwargs = {
            'num_stacks': 2,
            'cnv_dim': 256,
            'weights': self.weights,
        }
        heads = {
            'hm': 80,  # 3
            'reg': 2,  # 4
            'wh': 2  # 5
        }
        model = HourglassNetwork(heads=heads, **kwargs)
        self.model = CtDetDecode(model)
        self.letterbox_transformer = LetterboxTransformer(512, 512)

    def detect_image(self, image):
        img = pil2cv(image)
        pimg = self.letterbox_transformer(img)
        pimg = normalize_image(pimg)
        pimg = np.expand_dims(pimg, 0)
        objects = []
        detections = self.model.predict(pimg)[0]
        for d in detections:
            left, top, right, bottom, score, cl = d
            if score < 0.3:
                break
            left, top, right, bottom = self.letterbox_transformer.correct_box(left, top, right, bottom)
            cl = int(cl)
            predicted_class = coco_names[cl]
            objects.append({
                "bbox": [left, top, right, bottom],
                "score": np.asscalar(score),
                "class_name": predicted_class,
                "class_id": cl
            })

        return {
            "objects": objects
        }

    def close_session(self):
        pass
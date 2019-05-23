import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2
import colorsys

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
        hsv_tuples = [(x / len(coco_names), 1., 1.)
                      for x in range(len(coco_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
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
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
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
                "class": coco_names[cl],
            })
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[cl])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[cl])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        return image, objects

    def close_session(self):
        pass
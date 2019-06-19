import argparse
import ast
import itertools
import os

import cv2
import numpy as np
from scipy import ndimage
import random

import imgaug as ia
from imgaug import augmenters as iaa


def _list_to_str(lst):
    """Convert a list to string.

    Elements are separated by comma, without any spaces inserted in between
    """

    if type(lst) is not list:
        return str(lst)

    return '[' + ','.join(_list_to_str(elem) for elem in lst) + ']'

def rotate_point(x, y, angle):
    t = -np.radians(angle)  # radian
    nx = int(x * np.cos(t) - y * np.sin(t))
    ny = int(x * np.sin(t) + y * np.cos(t))
    return (nx, ny)


def flip_bbox_horizontal(img_w, img_h, angle):
    def f(bbox, modified_image, arg):
        tmp = bbox[0]
        bbox[0] = img_w - bbox[2]
        bbox[2] = img_w - tmp
        return bbox

    return f


def flip_bbox_vertical(img_w, img_h, angle):
    def f(bbox, modified_image, arg):
        tmp = bbox[1]
        bbox[1] = img_h - bbox[3]
        bbox[3] = img_h - tmp
        return bbox

    return f


def rotate_bbox_angle(img_w, img_h, angle):
    def f(bbox, modified_image, arg):
        x_pos, y_pos = [], []
        for x, y in itertools.product([bbox[0], bbox[2]], [bbox[1], bbox[3]]):
            nx, ny = rotate_point(x - img_w / 2, y - img_h / 2, angle)
            nx += modified_image.shape[1] // 2
            ny += modified_image.shape[0] // 2
            x_pos.append(nx)
            y_pos.append(ny)

        bbox[0] = min(x_pos)
        bbox[2] = max(x_pos)
        bbox[1] = min(y_pos)
        bbox[3] = max(y_pos)
        return bbox

    return f


def flip_polygon_horizontal(img_w, img_h, angle):
    def f(polygon, modified_image, arg):
        for i in range(0, len(polygon), 2):
            polygon[i] = img_w - polygon[i]
        return polygon

    return f


def flip_polygon_vertical(img_w, img_h, angle):
    def f(polygon, modified_image, arg):
        for i in range(1, len(polygon), 2):
            polygon[i] = img_h - polygon[i]
        return polygon

    return f


def rotate_polygon_angle(img_w, img_h, angle):
    def f(polygon, modified_image, arg):
        for i in range(0, len(polygon), 2):
            x, y = polygon[i], polygon[i + 1]
            nx, ny = rotate_point(x - img_w / 2, y - img_h / 2, angle)
            nx += modified_image.shape[1] // 2
            ny += modified_image.shape[0] // 2
            polygon[i], polygon[i + 1] = nx, ny
        return polygon

    return f

def make_random_sequential():
    seq = iaa.Sequential([
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        )
    ])
    seq_det = seq._to_deterministic()
    return seq_det

def make_noise_sequential():
    seqw = iaa.Sequential([

    ])


def random_bbox_transform(img_w, img_h, angle):
    def f(bbox, modified_image, seq_det):

        bbox_on_image = ia.BoundingBoxesOnImage.from_xyxy_array(np.float32([bbox]), shape=(img_h, img_w))

        bbox_on_image = seq_det.augment_bounding_boxes([bbox_on_image])[0]

        bbox = bbox_on_image.to_xyxy_array()[0].astype(np.int32).tolist()

        bbox[0] = max(0, min(img_w, bbox[0]))
        bbox[1] = max(0, min(img_h, bbox[1]))
        bbox[2] = max(0, min(img_w, bbox[2]))
        bbox[3] = max(0, min(img_h, bbox[3]))
        print(bbox)

        return bbox
    return f


def random_polygon_transform(img_w, img_h, angle):
    def f(polygon, modified_image, seq_det):

        poly = []
        for i in range(0, len(polygon), 2):
            poly.append( (polygon[i], polygon[i+1]) )
            """
            poly_x.append(polygon[i])
            poly_y.append(polygon[i + 1])
            """
            # rewrite for imgarg 0.2.8
        poly_on_image = ia.PolygonsOnImage([ia.Polygon(poly)], shape=(img_h, img_w))
        poly_on_image = seq_det.augment_polygons([poly_on_image])[0]

        moved_poly = poly_on_image.polygons[0]
        for i, (x, y) in enumerate(zip(moved_poly.xx, moved_poly.yy)):
            polygon[2 * i] = x
            polygon[2 * i + 1] = y

        return polygon
    return f


def rotate_image(angle, image):
    if angle == 0:
        return image
    elif angle == 180:
        return cv2.flip(image, -1)

    h, w, _ = image.shape
    t = np.radians(angle)
    h_ = abs(h * np.cos(t) + w * np.sin(t))  # rotate image size
    w_ = abs(w * np.cos(t) + h * np.sin(t))

    image_top = image[:1, :, :]
    image_top = cv2.resize(image_top, (w, w))
    image_bottom = image[h - 1:, :, :]
    image_bottom = cv2.resize(image_bottom, (w, w))
    image_center = cv2.vconcat([image_top, image, image_bottom])

    blank_shape = (w, h, 3)
    blank = np.zeros(blank_shape, dtype=np.uint8)

    image_left = image[:, :1, :]
    image_left = cv2.resize(image_left, (h, h))
    image_left = cv2.vconcat([blank, image_left, blank])

    image_right = image[:, w - 1:, :]
    image_right = cv2.resize(image_right, (h, h))
    image_right = cv2.vconcat([blank, image_right, blank])

    image = cv2.hconcat([image_left, image_center, image_right])
    image = ndimage.rotate(image, angle, reshape=True)
    bh, bw, _ = image.shape
    image = image[int((bh - h_) / 2):int((bh + h_) / 2),
                  int((bw - w_) / 2):int((bw + w_) / 2), :]
    return image


def change_object_color(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    r = random.randint(0, 255)
    img[:,:,0] += r
    img[:,:,0] %= 256
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    res_img = img.copy()

    res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2HSV)
    r = random.randint(0, 255)
    res_img[:,:,0] += r
    res_img[:,:,0] %= 256
    res_img = cv2.cvtColor(res_img, cv2.COLOR_HSV2BGR)
    res_img = np.where(mask==255, res_img, img)
    return res_img


def rotate_image_and_annotation(img,
                                areas,
                                bbox_transform_function,
                                class_names,
                                object_type=None,
                                horizontal=False,
                                vertical=False,
                                angle=None,
                                affin=False,
                                trans_color=False):
    arg = None
    if horizontal:
        rotated_img = cv2.flip(img, 1)
    elif vertical:
        rotated_img = cv2.flip(img, 0)
    elif angle is not None:
        rotated_img = rotate_image(angle, img)
    else:
        arg = make_random_sequential()
        rotated_img = arg.augment_images([img.copy()])[0]


    annotation = []

    img_mask = img = np.zeros(list(img.shape), dtype=np.uint8)
    box_list = []
    class_list = []
    for obj_area in areas:

        if "[" in obj_area:
            assert object_type == "polygon"

            boxes, class_id = ast.literal_eval(obj_area)
            boxes = [
                bbox_transform_function(box, rotated_img, arg) for box in boxes
            ]
            for box in boxes:
                poly = np.array([(x, y) for x, y in zip(box[::2], box[1::2])]).astype(np.int32)
                cv2.fillPoly(
                    img_mask,
                    pts = [poly],
                    color = (255, 255, 255))
            box_list.append(boxes)
        else:
            *box, class_id = map(int, obj_area.split(','))
            box = bbox_transform_function(box, rotated_img, arg)
            if object_type == "polygon":
                poly = np.array([(x, y) for x, y in zip(box[::2], box[1::2])]).astype(np.int32)
                cv2.fillPoly(
                    img_mask,
                    pts = [poly],
                    color = (255, 255, 255))
                boxes = [box]
                del box
                box_list.append(boxes)
            else:
                cv2.rectangle(
                    img_mask, (box[0], box[1]), (box[2], box[3]),
                    (255, 255, 255),
                    thickness=-1,
                    lineType=cv2.LINE_AA)
                box_list.append(box)
        class_list.append(class_id)

    if trans_color:
        rotated_img = change_object_color(rotated_img, img_mask)
    annotated_img = rotated_img.copy()

    for obj, class_id in zip(box_list, class_list):
        if object_type == "bbox":
            box = obj
            cv2.rectangle(
                annotated_img, (box[0], box[1]), (box[2], box[3]),
                (255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA)
            cv2.putText(annotated_img, class_names[class_id], (box[0], box[1]),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            annotation.append((box, class_id))
        else:
            boxes = obj
            cv2.polylines(
                annotated_img,
                [np.reshape(box, (-1, 2)).astype(np.int32) for box in boxes],
                isClosed=True,
                color=(255, 0, 0),
                thickness=2,
            )
            for box in boxes:
                cv2.putText(annotated_img, class_names[class_id],
                            (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
            annotation.append((boxes, class_id))

    return rotated_img, annotated_img, annotation


def rotate(annotations_path,
           horizontal=None,
           vertical=None,
           affine=None,
           angle=None,
           trans_color=False,
           numbers_of_trans=100):

    horizontal = bool(horizontal)
    vertical = bool(vertical)
    affine = bool(affine)
    if angle is not None:
        angle = int(angle)

    basedir = os.path.dirname(annotations_path)

    with open(annotations_path) as f:
        annotations = [f_line.split() for f_line in f][:numbers_of_trans]


    if len(annotations[0][1].split(',')) == 5:
        object_type = 'bbox'
    else:
        object_type = 'polygon'

    if horizontal:
        suf = "_h"
        bbox_transform_function_factory = flip_bbox_horizontal if object_type == 'bbox' else flip_polygon_horizontal
    elif vertical:
        suf = "_v"
        bbox_transform_function_factory = flip_bbox_vertical if object_type == 'bbox' else flip_polygon_vertical
    elif angle is not None:
        suf = "_a" + str(angle)
        bbox_transform_function_factory = rotate_bbox_angle if object_type == 'bbox' else rotate_polygon_angle
    else:
        suf = "_t"
        bbox_transform_function_factory = random_bbox_transform if object_type == 'bbox' else random_polygon_transform

    classes_path = os.path.join(basedir, "classes.txt")
    new_annotations_path = suf.join(os.path.splitext(annotations_path))
    with open(classes_path) as f:
        class_names = [c.strip() for c in f]

    annotations_ = []
    for i, data in enumerate(annotations):
        img_file, areas = data[0], data[1:]

        img_file_name = os.path.basename(img_file)

        img = cv2.imread(img_file)

        if bbox_transform_function_factory is not None:
            bbox_transform_function = bbox_transform_function_factory(
                img.shape[1], img.shape[0], angle)


        (rotated_img, annotated_img,
         annotation) = rotate_image_and_annotation(
             img, areas, bbox_transform_function, class_names, object_type,
             horizontal, vertical, angle, affine, trans_color)

        new_img_dir = os.path.dirname(img_file) + suf
        os.makedirs(new_img_dir, exist_ok=True)
        img_new_file = os.path.join(new_img_dir, img_file_name)
        cv2.imwrite(img_new_file, rotated_img)

        print(img_new_file)
        annotations_.append((img_new_file, annotation))

        verbose_dir = os.path.dirname(img_file) + suf + "_annot"
        os.makedirs(verbose_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(verbose_dir, img_file_name), annotated_img)

    with open(new_annotations_path, mode='w') as f:
        for img_filepath, annotation in annotations_:
            if object_type == "bbox":
                print(
                    img_filepath,
                    " ".join("{box},{class_id}".format(
                        box=",".join(map(str, box)),
                        class_id=class_id,
                    ) for box, class_id in annotation),
                    file=f,
                )
            else:
                print(
                    img_filepath,
                    " ".join("{polygons},{class_id}".format(
                        polygons=_list_to_str(polygons),
                        class_id=class_id,
                    ) for polygons, class_id in annotation),
                    file=f,
                )

    print(new_img_dir)
    print(new_annotations_path)


def _main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('-a', '--annotations_path', required=True)
    group.add_argument('-ho', '--horizontal', action='store_true')
    group.add_argument('-ve', '--vertical', action='store_true')
    group.add_argument('-ang', '--angle', type=int)
    group.add_argument('-af', '--affine_transform', action='store_true')
    parser.add_argument('-c', '--trans_color', action='store_true')
    parser.add_argument('-n', '--numbers_of_trans', type=int, default=100)

    args = parser.parse_args()
    rotate(
        args.annotations_path,
        horizontal=args.horizontal,
        vertical=args.vertical,
        affine=args.affine_transform,
        angle=args.angle,
        trans_color=args.trans_color,
        numbers_of_trans=args.numbers_of_trans
    )


if __name__ == '__main__':
    _main()

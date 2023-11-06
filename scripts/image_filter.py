import argparse
import cv2
import os
import numpy as np
from lib.utils import *

from lib.Evaluator import *
from lib.utils import BBFormat
from size_histogram import get_bboxes_and_classes


def sobel_horizontal(image):
    kernel_sobel_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    image_output = cv2.filter2D(image, cv2.CV_64F, kernel_sobel_h)
    return image_output


def sobel_vertical(image):
    kernel_sobel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    image_output = cv2.filter2D(image, cv2.CV_64F, kernel_sobel_v)
    return image_output


def sobel(image):
    image_h = sobel_horizontal(image)
    image_v = sobel_vertical(image)
    image_output = np.sqrt(image_h**2 + image_v**2)
    return image_output


def save_filter_applied_images(ground_truth_dir_path, all_bboxes, filter, ratio):
    save_dir_path = os.path.join(f"images/{filter}")
    os.makedirs(save_dir_path, exist_ok=True)

    classes_file_path = os.path.join(ground_truth_dir_path, "../classes.txt")
    with open(classes_file_path) as f:
        class_names = [line.strip() for line in f]

    test_file_path = os.path.join(ground_truth_dir_path, "../test.txt")
    with open(test_file_path) as f:
        image_file_path_list = [line.strip().split()[0] for line in f]
    originalvar = {}
    extendedvar = {}
    originalmean = {}
    extendedmean = {}
    for a in class_names:
        originalvar[a] = np.empty(0)
        extendedvar[a] = np.empty(0)
        originalmean[a] = np.empty(0)
        extendedmean[a] = np.empty(0)
    for image_file_path in image_file_path_list:
        image_file_name = os.path.basename(image_file_path)
        image_file_name_witout_ext = os.path.splitext(image_file_name)[0]

        image = cv2.imread(image_file_path)
        bboxes = all_bboxes.getBoundingBoxesByImageName(image_file_name_witout_ext)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if filter == "sobel":
            image = sobel(image)
        elif filter == "laplacian":
            image = cv2.Laplacian(image, cv2.CV_32F)

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.getAbsoluteBoundingBox(BBFormat.XYX2Y2)

            class_id = bbox.getClassId()
            confidence = str(bbox.getConfidence())
            ex1 = int(x1 - (x2 - x1) * ratio)
            ey1 = int(y1 - (y2 - y1) * ratio)
            ex2 = int(x2 + (x2 - x1) * ratio)
            ey2 = int(y2 + (y2 - y1) * ratio)
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            if ex1 < 0:
                ex1 = 0
            if ex2 > image.shape[1]:
                ex2 = image.shape[1] - 1
            if ey1 < 0:
                ey1 = 0
            if ey2 > image.shape[0]:
                ey2 = image.shape[0] - 1
            e1 = image[ey1:y1, ex1:x2]
            e2 = image[ey1:y2, x2:ex2]
            e3 = image[y2:ey2, x1:ex2]
            e4 = image[y1:ey2, ex1:x1]
            pred = image[y1:y2, x1:x2]
            e1 = np.ravel(e1)
            e2 = np.ravel(e2)
            e3 = np.ravel(e3)
            e4 = np.ravel(e4)
            e = np.concatenate([e1, e2, e3, e4])
            originalvar[class_id] = np.append(originalvar[class_id], pred.var())
            originalmean[class_id] = np.append(originalmean[class_id], pred.mean())
            extendedvar[class_id] = np.append(extendedvar[class_id], e.var())
            extendedmean[class_id] = np.append(extendedmean[class_id], e.mean())

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
            class_id = bbox.getClassId()
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            if confidence != "None":
                continue
            cv2.rectangle(
                image,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                image,
                class_id,
                (x1, y1),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        save_image_file_path = os.path.join(save_dir_path, image_file_name)
        cv2.imwrite(save_image_file_path, image)

    print("original var")
    for a in class_names:
        print(f"{a}:{round(np.average(originalvar[a]),2)}")

    print("extended var")
    for a in class_names:
        print(f"{a}:{round(np.average(extendedvar[a]),2)}")

    print("original mean")
    for a in class_names:
        print(f"{a}:{round(np.average(originalmean[a]),2)}")

    print("extended var")
    for a in class_names:
        print(f"{a}:{round(np.average(extendedmean[a]),2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--ground_truth_dir_path",
        type=str,
        required=True,
        help="path to the ground-truth directory path",
    )
    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        choices=[
            "sobel",
            "laplacian",
        ],
        help="filter name(sobel,Laplacian)",
        default="gray",
    )
    parser.add_argument(
        "-r",
        "--ratio",
        type=float,
        help="extend ratio",
        default=0.1,
    )
    args = vars(parser.parse_args())

    bboxes = get_bboxes_and_classes(args["ground_truth_dir_path"])
    save_filter_applied_images(
        args["ground_truth_dir_path"], bboxes, args["filter"], args["ratio"]
    )

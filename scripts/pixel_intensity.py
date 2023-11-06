import argparse
import os
import numpy as np
import glob
from lib.utils import *
import colorsys


from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import BBFormat, CoordinatesType, BBType


def generate_colors(class_num):
    hsv_tuples = [(x / class_num, 1.0, 1.0) for x in range(class_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (float(x[0] * 1), float(x[1] * 1), float(x[2] * 1)), colors)
    )
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors


def get_bboxes_and_classes(ground_truth_dir_path):
    gt_dict = {}
    for file_path in glob.glob(os.path.join(ground_truth_dir_path, "*.txt")):
        image_name = os.path.splitext(os.path.basename(file_path))[0]
        gt_dict[image_name] = []
        with open(file_path, "r") as f:
            for line in f:
                obj_dict = {}
                class_name, sx, sy, ex, ey = line.split("\t")
                obj_dict["class_name"] = class_name
                obj_dict["bbox"] = [float(sx), float(sy), float(ex), float(ey)]
                gt_dict[image_name].append(obj_dict)

    allBoundingBoxes = BoundingBoxes()

    for img_filename, prediction in gt_dict.items():
        # BBox of groundTruth
        true_annotation = gt_dict[img_filename]
        for obj in true_annotation:
            bbox = obj["bbox"]
            class_name = obj["class_name"]
            bb = BoundingBox(
                img_filename,
                class_name,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                CoordinatesType.Absolute,
                None,
                BBType.GroundTruth,
                format=BBFormat.XYX2Y2,
            )
            allBoundingBoxes.addBoundingBox(bb)

    return allBoundingBoxes


def plot_graph_size(boundingboxes, test_file_path):
    savePath = "figs"
    groundTruths = []
    classes = []

    os.makedirs(savePath, exist_ok=True)
    for bb in boundingboxes.getBoundingBoxes():
        if bb.getBBType() == BBType.GroundTruth:
            groundTruths.append(
                [
                    bb.getImageName(),
                    bb.getClassId(),
                    1,
                    bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2),
                ]
            )
        if bb.getClassId() not in classes:
            classes.append(bb.getClassId())
    classes = sorted(classes)
    color = generate_colors(len(classes))
    with open(test_file_path) as f:
        image_file_path_list = [line.strip().split()[0] for line in f]
    array = {}
    images = {}
    n = {}
    for a in classes:
        array[a] = np.empty(0)
        images[a] = []
        n[a] = 0
    for image_file_path in image_file_path_list:
        image_file_name = os.path.basename(image_file_path)
        image_file_name_witout_ext = os.path.splitext(image_file_name)[0]
        bboxes = boundingboxes.getBoundingBoxesByImageName(image_file_name_witout_ext)
        image = cv2.imread(image_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            img = image[y1:y2,x1:x2]
            images[bbox.getClassId()].extend(img.reshape(-1))
            n[bbox.getClassId()] += 1 
    ax = []
    fig = plt.figure(figsize=(15, 8))
    for a, i in zip(classes, range(len(classes))):
        ax.append(fig.add_subplot(2, 4, i + 1))
    for a, i in zip(classes, range(len(classes))):
        ax[i].hist(images[a],range(0,255), weights=np.ones(len(images[a]))/n[a], color=color[i])
        ax[i].set_title(a)
    
    fig.supxlabel("Pixel Intensity", fontsize=20)
    fig.supylabel("Frequency", fontsize=20)

    plt.subplots_adjust(right=0.93)
    plt.subplots_adjust(left=0.07)
    plt.show()
    fig.savefig(savePath + "/pixel_intensity_hist")


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
        "-t",
        "--test_file_path",
        type=str,
        required=True,
        help="path to test file path",
    )
    args = vars(parser.parse_args())

    bboxes = get_bboxes_and_classes(args["ground_truth_dir_path"])
    plot_graph_size(bboxes, args["test_file_path"])

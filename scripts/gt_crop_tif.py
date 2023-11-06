import argparse
import os
from lib.utils import *

from lib.Evaluator import *
from lib.utils import BBFormat, BBType
from size_histogram import get_bboxes_and_classes


def crop(boundingboxes, test_file_path):
    savePath = "crop"
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
    for a in classes:
        os.makedirs(savePath + "/" + a, exist_ok=True)
    with open(test_file_path) as f:
        image_file_path_list = [line.strip().split()[0] for line in f]
    for image_file_path in image_file_path_list:
        num = 0
        image_file_name = os.path.basename(image_file_path)
        image_file_name_witout_ext = os.path.splitext(image_file_name)[0]
        bboxes = boundingboxes.getBoundingBoxesByImageName(image_file_name_witout_ext)
        image = cv2.imread(image_file_path)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            saveimg = image[y1:y2, x1:x2]
            cv2.imwrite(
                f"{savePath}/{bbox.getClassId()}/{image_file_name_witout_ext}_{str(num)}.tiff",
                saveimg,
            )
            num += 1


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
        help="path to the test.txt path",
    )
    args = vars(parser.parse_args())
    bboxes = get_bboxes_and_classes(ground_truth_dir_path=args["ground_truth_dir_path"])
    crop(bboxes, args["test_file_path"])

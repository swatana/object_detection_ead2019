import argparse
import cv2
import os
import numpy as np
import sys
import glob
from utils import get_annotations

"""clone repository 'https://github.com/seqsense/Object-Detection-Metrics'"""

from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import BBFormat, CoordinatesType, BBType, MethodAveragePrecision

def non_maximum_suppression(classes, boxes, scores, ios_threshold, iou_threshold):
    """Non Maximum Suppression"""

    boxes_by_class = {}
    scores_by_class = {}

    selected_classes = []
    selected_boxes = []
    selected_scores = []

    for c, b, s in zip(classes, boxes, scores):
        if(c in boxes_by_class):
            boxes_by_class[c].append(b)
            scores_by_class[c].append(s)
        else:
            boxes_by_class[c] = [b]
            scores_by_class[c] = [s]

    for class_id in boxes_by_class:
        boxes = np.array(boxes_by_class[class_id])
        scores = np.array(scores_by_class[class_id])
        x1, y1, x2, y2 = np.split(boxes, 4, axis=1)

        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        indices = np.argsort(scores)
        selected = []

        while len(indices) > 0:
            last = len(indices) - 1

            selected_index = indices[last]
            remaining_indices = indices[:last]
            selected.append(selected_index)

            i_x1 = np.maximum(x1[selected_index], x1[remaining_indices])
            i_y1 = np.maximum(y1[selected_index], y1[remaining_indices])
            i_x2 = np.minimum(x2[selected_index], x2[remaining_indices])
            i_y2 = np.minimum(y2[selected_index], y2[remaining_indices])

            i_w = np.maximum(0, i_x2 - i_x1 + 1)
            i_h = np.maximum(0, i_y2 - i_y1 + 1)
            overlap_area = i_w * i_h

            # suppression by IoS
            overlap = np.maximum(overlap_area / area[remaining_indices], overlap_area / area[selected_index])
            delete_indices_ios = np.where(overlap > ios_threshold)[0]

            # suppression by IoU
            overlap = overlap_area / (area[selected_index] + area[remaining_indices] - overlap_area)
            delete_indices_iou = np.where(overlap > iou_threshold)[0]

            delete_indices = np.concatenate((delete_indices_iou, delete_indices_ios))
            indices = np.delete(
                indices,
                np.concatenate(([last], np.unique(delete_indices)))
            )

        selected_classes.extend(np.full(len(selected), class_id).tolist())
        selected_boxes.extend(boxes[selected].tolist())
        selected_scores.extend(scores[selected].tolist())

    return (np.array(selected_classes), np.array(selected_boxes), np.array(selected_scores))

def get_bboxes_and_classes(ground_truth_dir_path, prediction_dir_path, score_threshold, ios_threshold, iou_threshold):

    gt_dict = {}
    for file_path in glob.glob(os.path.join(ground_truth_dir_path, '*.txt')):
        image_name = os.path.splitext(os.path.basename(file_path))[0]
        gt_dict[image_name] = []
        with open(file_path, 'r') as f:
            for line in f:
                obj_dict = {}
                class_name, sx, sy, ex, ey = line.split('\t')
                obj_dict['class_name'] = class_name
                obj_dict['bbox'] = [float(sx), float(sy), float(ex), float(ey)]
                gt_dict[image_name].append(obj_dict)

    predictions_dict = {}
    for file_path in glob.glob(os.path.join(prediction_dir_path, '*.txt')):
        image_name = os.path.splitext(os.path.basename(file_path))[0]
        predictions_dict[image_name] = []
        with open(file_path, 'r') as f:
            for line in f:
                obj_dict = {}
                class_name, score, sx, sy, ex, ey = line.split('\t')
                obj_dict['class_name'] = class_name
                obj_dict['score'] = float(score)
                obj_dict['bbox'] = [float(sx), float(sy), float(ex), float(ey)]
                predictions_dict[image_name].append(obj_dict)

    allBoundingBoxes = BoundingBoxes()

    for img_filename, prediction in predictions_dict.items():

        # BBox of groundTruth
        true_annotation = gt_dict[img_filename]
        for obj in true_annotation:
            bbox = obj["bbox"]
            class_name = obj['class_name']
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
                format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)

        # Non Maximum Suppression of predictions
        classes = []
        boxes = []
        scores = []
        for obj in prediction:
            if obj["score"] >= score_threshold:
                classes.append(obj["class_name"])
                boxes.append(obj["bbox"])
                scores.append(obj["score"])
        
        classes, boxes, scores = non_maximum_suppression(classes, boxes, scores, ios_threshold, iou_threshold)
        for class_name, bbox, score in zip(classes, boxes, scores):
            bb = BoundingBox(
                img_filename,
                class_name,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                CoordinatesType.Absolute,
                None,
                BBType.Detected,
                score,
                format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)

    return allBoundingBoxes


def plot_graph(allBoundingBoxes, savePath, thresholds):
    IOU_THRESHOLD = 0.25 # default value
    evaluator = Evaluator()
    acc_AP = 0
    acc_IoU = 0
    validClasses = 0

    savePath = os.path.join(savePath, "mAP_IoU_" + "_".join([str(thre) for thre in thresholds]))
    os.makedirs(savePath,  exist_ok=True)
    os.makedirs(os.path.join(savePath, "graphs"),  exist_ok=True)

    # Plot Precision x Recall curve
    detections = evaluator.PlotPrecisionRecallCurve(
        allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=IOU_THRESHOLD,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
        savePath=os.path.join(savePath, "graphs"),
        showGraphic=False)

    with open(os.path.join(savePath, 'results.txt'), 'w') as f:
        f.write('Object Detection Metrics\n')
        f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
        f.write('# AP and precision/recall per class')

        # each detection is a class
        for metricsPerClass in detections:

            # Get metric values per each class
            cl = metricsPerClass['class']
            ap = metricsPerClass['AP']
            iou = metricsPerClass['IoU']
            precision = metricsPerClass['precision']
            recall = metricsPerClass['recall']
            totalPositives = metricsPerClass['total positives']

            if totalPositives > 0:
                validClasses = validClasses + 1
                acc_AP = acc_AP + ap
                acc_IoU = acc_IoU + iou
                prec = ['%.2f' % p for p in precision]
                rec = ['%.2f' % r for r in recall]
                ap_str = "{0:.2f}%".format(ap * 100)
                iou_str = "{0:.2f}%".format(iou * 100)
                print('AP: %s (%s)' % (ap_str, cl))
                print('IoU: %s (%s)' % (iou_str, cl))
                f.write('\n\nClass: %s' % cl)
                f.write('\nAP: %s' % ap_str)
                f.write('\nIoU: %s' % iou_str)
                f.write('\nPrecision: %s' % prec)
                f.write('\nRecall: %s' % rec)

        mAP = acc_AP / validClasses
        mAP_str = "{0:.2f}%".format(mAP * 100)
        print('mAP: %s' % mAP_str)
        mIoU = acc_IoU / validClasses
        mIoU_str = "{0:.2f}%".format(mIoU * 100)
        print('mIoU: %s' % mIoU_str)
        f.write('\n\n# mAP of all classes\nmAP: %s\nmIoU: %s' % (mAP_str, mIoU_str))


        f.write("\n\n# Number of ground-truth objects per class")
        for metricsPerClass in detections:

            cl = metricsPerClass['class']
            totalPositives = metricsPerClass['total positives']

            if totalPositives > 0:
                f.write('\n%s: %d' % (cl, totalPositives))


        f.write("\n\n# Number of predicted objects per class")
        for metricsPerClass in detections:

            cl = metricsPerClass['class']
            totalPositives = metricsPerClass['total positives']
            total_TP = int(metricsPerClass['total TP'])
            total_FP = int(metricsPerClass['total FP'])

            if totalPositives > 0:
                f.write('\n%s: %d (tp:%d, fp: %d)' % (cl, total_TP + total_FP, total_TP, total_FP))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g', '--ground_truth_dir_path', type=str, required=True,
        help='path to the ground-truth directory path'
    )
    parser.add_argument(
        '-r', '--result_dir_path', type=str, required=True,
        help='path to the result directory path'
    )
    parser.add_argument(
        '-th', '--score_threshold', type=float, default=0.01,
        help='Score threshold'
    )
    parser.add_argument(
        '-iou', '--iou_threshold', type=float, default=0.3,
        help='Non maximum suppression threshold for Intersection over Union'
    )
    parser.add_argument(
        '-ios', '--ios_threshold', type=float, default=1,
        help='Non maximum suppression threshold for Intersection over Section'
    )
    args = vars(parser.parse_args())
    
    bboxes = get_bboxes_and_classes(
                            ground_truth_dir_path=args["ground_truth_dir_path"],
                            prediction_dir_path=os.path.join(args["result_dir_path"], 'predictions'),
                            score_threshold=args["score_threshold"],
                            iou_threshold=args["iou_threshold"],
                            ios_threshold=args["ios_threshold"])
    plot_graph(bboxes, args["result_dir_path"], [args["score_threshold"], args["iou_threshold"], args["ios_threshold"]])

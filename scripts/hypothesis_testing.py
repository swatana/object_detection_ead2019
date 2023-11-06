from scipy import stats
import scikit_posthocs as sp
import pandas as pd
import argparse
import cv2
import os
import numpy as np
import math
import csv
from lib.utils import *
from lib.Evaluator import *
from lib.utils import BBFormat
from size_histogram import get_bboxes_and_classes

import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


graph_colors={
    'specularity': '#ff0000',
    'saturation': '#4900ff',
    'artifact': '#ff00db',
    'blur': '#00ff92',
    'contrast': '#ffdb00',
    'bubbles': '#0092ff',
    'instrument': '#49ff00',
}

def save_filter_applied_images(ground_truth_dir_path, all_bboxes, ratio):

    classes_file_path = os.path.join(ground_truth_dir_path, "../classes.txt")
    with open(classes_file_path) as f:
        class_names = [line.strip() for line in f]

    test_file_path = os.path.join(ground_truth_dir_path, "../test.txt")
    with open(test_file_path) as f:
        image_file_path_list = [line.strip().split()[0] for line in f]
    diff = {}
    for a in class_names:
        diff[a] = np.empty(0)
    for image_file_path in image_file_path_list:
        image_file_name = os.path.basename(image_file_path)
        image_file_name_witout_ext = os.path.splitext(image_file_name)[0]

        image = cv2.imread(image_file_path)
        bboxes = all_bboxes.getBoundingBoxesByImageName(image_file_name_witout_ext)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.getAbsoluteBoundingBox(BBFormat.XYX2Y2)

            class_id = bbox.getClassId()
            confidence = str(bbox.getConfidence())
            ex1 = int(x1 - (x2 - x1) * ratio / 2)
            ey1 = int(y1 - (y2 - y1) * ratio / 2)
            ex2 = int(x2 + (x2 - x1) * ratio / 2)
            ey2 = int(y2 + (y2 - y1) * ratio / 2)
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
            sqrt = math.sqrt((pred.mean() - e.mean())**2)
            if (not np.isnan(sqrt)):
                diff[class_id] = np.append(diff[class_id], sqrt)
            else:
                print(f'skip {image_file_path} {class_id} {x1} {y1} {x2} {y2} {image.shape}')

    array = list(diff.values())
    s, p = stats.kruskal(array[0], array[1], array[2], array[3], array[4], array[5], array[6])
    print("kruskal statistic: " + str(s))
    print("kruskal pvalue: " + str(p))
    graph_data = {}
    graph_array = []
    for a in class_names:
        graph_data[a] = np.average(diff[a])
        graph_array.append(graph_data[a])
        print(a + " average: " + str(np.average(diff[a])))
    print(sp.posthoc_dscf(array))
    graph_average = np.average(graph_array)
    array= np.array(list(graph_data.values())).reshape(-1,1)
    color=[graph_colors[a] for a in graph_data]
    fig, ax = plt.subplots(figsize = (12,10))
    ax.bar(graph_data.keys(), graph_data.values(),color=color)
    plt.ylabel('Difference in Average Intensity', fontsize=20)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45, fontsize=15)
    ratiokey = f'{int(ratio * 100):03d}'
    # fig.savefig(f"figs/hypothesis_{ratiokey}.png")
    # KMeans
    km = KMeans(n_clusters=2, random_state=42)
    km = km.fit(array)
    centroids = km.cluster_centers_
    distances = abs(centroids[0][0] - centroids[1][0])
    # export to json
    output = os.path.join(os.path.dirname(__file__), f'{os.path.splitext(os.path.basename(__file__))[0]}.json')
    current = {}
    if os.path.isfile(output):
        with open(output, 'r', encoding='utf-8') as f:
            current = json.load(f)
    current[ratiokey] = [distances, graph_average, distances / graph_average]
    current = dict(sorted(current.items()))
    with open(output, 'w', encoding='utf-8') as result:
        json.dump(current, result, ensure_ascii=False, indent=2)
    # export to image
    # match chart colors
    if centroids[0][0] - centroids[1][0] >= 0:
        colors = ["red", "blue"]
    else:
        colors = ["blue", "red"]
    color = [colors[a] for a in km.labels_]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.bar(graph_data.keys(), graph_data.values(), color=color)
    plt.ylabel('Difference in Average Intensity', fontsize=20)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45, fontsize=15)
    fig.savefig(f"figs/hypothesis_{ratiokey}_k_means.png")
    # export to csv
    output_csv = os.path.join(os.path.dirname(__file__), f'{os.path.splitext(os.path.basename(__file__))[0]}.csv')
    with open(output_csv, 'a', encoding='utf-8') as f:
        value = list(graph_data.values())
        value.insert(0, "{0}%".format(int(ratio * 100)))
        writer = csv.writer(f)
        writer.writerow(value)


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
        "-r",
        "--ratio",
        type=float,
        help="extend ratio",
        default=0.1,
    )
    args = vars(parser.parse_args())

    bboxes = get_bboxes_and_classes(args["ground_truth_dir_path"])
    save_filter_applied_images(
        args["ground_truth_dir_path"], bboxes, args["ratio"]
    )



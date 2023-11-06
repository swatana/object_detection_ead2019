import os
import argparse
from sklearn.model_selection import train_test_split
from random import shuffle, seed
from collections import defaultdict
import shutil
import sys

class NotFoundError(Exception):
    pass


def get_unused_dir_num(pdir, pref=None):
    print(pdir)
    print(pref)
    os.makedirs(pdir, exist_ok=True)
    dir_list = os.listdir(pdir)
    for i in range(1000):
        search_dir_name = "" if pref is None else (pref + "_") + '%03d' % i
        if search_dir_name not in dir_list:
            return os.path.join(pdir, search_dir_name)
    raise NotFoundError('Error')


def select_sample(items, lower_bound, test_ratio):
    print(len(items))
    train_items = []
    test_items = []
    
    seed(42)
    shuffle(items)
    class_counts = defaultdict(int)
    for filename, class_ids in items:
        is_test = False
        for class_id in class_ids:
            is_test |= (class_counts[class_id] < lower_bound)
        if is_test:
            test_items.append(filename)
            for class_id in class_ids:
                class_counts[class_id] += 1
        else:
            train_items.append(filename)

    idx = 0
    if test_ratio:
        while idx < len(train_items) and len(test_items) <= test_ratio * len(items):
            test_items.append(train_items[idx])
            idx += 1

    return train_items[idx:], test_items


def split_annotation(annotation_file_path, annotation_file_format, classes_file_path, lower_bound=None, test_ratio=0.2): # "data_labels/coco_fire"
    annotation_dir = os.path.dirname(annotation_file_path)
    dataset_name = os.path.basename(annotation_dir)
    upper_annnotation_dir = os.path.dirname(annotation_dir)
    if upper_annnotation_dir == '':
        upper_annnotation_dir = '.'
    output_dir = get_unused_dir_num(upper_annnotation_dir, dataset_name)
    with open(classes_file_path) as f:
        class_list = [line.strip() for line in f]
    with open(annotation_file_path) as f:
        annotation_lines = [line.strip() for line in f]

    os.makedirs(output_dir, exist_ok=True)

    shutil.copyfile(classes_file_path, os.path.join(output_dir, "classes.txt"))

    if lower_bound:
        image_file_path_2_annotation_line = dict()
        train_items = []
        for item in annotation_lines:
            image_file_path, *objects = item.split()
            image_file_path_2_annotation_line[image_file_path] = item
            class_ids = [int(obj.split(',')[-1].strip(']')) for obj in objects]
            train_items.append((image_file_path, class_ids))

        train_image_file_path_list, test_image_file_path_list = select_sample(train_items, lower_bound, test_ratio)

        train_annotation_lines = [image_file_path_2_annotation_line[train_image_file_path] for train_image_file_path in train_image_file_path_list]
        test_annotation_lines = [image_file_path_2_annotation_line[test_image_file_path] for test_image_file_path in test_image_file_path_list]
    elif test_ratio == 0:
        train_annotation_lines = annotation_lines
        test_annotation_lines = []
    elif test_ratio == 1:
        train_annotation_lines = []
        test_annotation_lines = annotation_lines
    else:
        train_annotation_lines, test_annotation_lines = train_test_split(
            annotation_lines, test_size=test_ratio, random_state=42)

    print(f"{len(train_annotation_lines)} train annotations and {len(test_annotation_lines)} test annotations")
    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        train_annotation_output = "\n".join(train_annotation_lines)
        f.write(train_annotation_output)

    with open(os.path.join(output_dir, "test.txt"), "w") as f:
        test_annotation_output = "\n".join(test_annotation_lines)
        f.write(test_annotation_output)

    print(f"Saved train.txt and test.txt in {output_dir}")
    if annotation_file_format != 'bbox':
        return

    ground_truth_output_dir = os.path.join(output_dir, "ground-truth")
    os.makedirs(ground_truth_output_dir, exist_ok=True)

    for test_annotation_line in test_annotation_lines:
        test_annotation_line_objects = test_annotation_line.split()
        image_file_path = test_annotation_line_objects[0]
        bboxes = test_annotation_line_objects[1:]

        image_file_basename = os.path.basename(image_file_path)
        image_file_basename_without_ext, _ = os.path.splitext(image_file_basename)

        ground_truth_file_output_lines = []
        for bbox in bboxes:
            bbox_objects = bbox.split(",")
            x_min = bbox_objects[0]
            y_min = bbox_objects[1]
            x_max = bbox_objects[2]
            y_max = bbox_objects[3]
            class_id = int(bbox_objects[4])
            class_name = class_list[class_id]
            print(class_id)
            print(class_list[class_id])

            ground_truth_file_output_lines.append(f"{class_name}\t{x_min}\t{y_min}\t{x_max}\t{y_max}")

        with open(os.path.join(ground_truth_output_dir, image_file_basename_without_ext + ".txt"), "w") as f:
            ground_truth_file_output = "\n".join(ground_truth_file_output_lines)
            f.write(ground_truth_file_output)

    print(f"Saved ground truth files in {ground_truth_output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--annotation_file_path', type=str, default=None, required=True,
        help='path to annotation file'
    )

    parser.add_argument(
        '-af',
        '--annotation_file_format',
        type=str,
        choices=[
            'bbox',
            'polygon',
            'tif',
        ],
        help='Annotation file format.',
        required=True,
    )

    parser.add_argument(
        '-c', '--classes_file_path', type=str,
        help='path to class definitions',
        required=True,
    )

    parser.add_argument(
        '-r', '--test_ratio', type=float,
        help='ratio of test annotations'
    )

    parser.add_argument(
        "-n",
        "--lower_bound",
        help="The least number of apperance of each class. Available except tif.",
        type=int
    )

    args = vars(parser.parse_args())

    if(args["annotation_file_format"] == "tif" and args["lower_bound"] is not None):
        sys.exit("You cannot specify lowerbound when the annotation file format is tif.")

    split_annotation(annotation_file_path=args["annotation_file_path"], annotation_file_format=args['annotation_file_format'], classes_file_path=args["classes_file_path"], lower_bound=args["lower_bound"], test_ratio=args["test_ratio"])

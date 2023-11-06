import os
import argparse
from sklearn.model_selection import KFold
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
        search_dir_name = "" if pref is None else (pref + "_kfold_") + '%03d' % i
        if search_dir_name not in dir_list:
            return os.path.join(pdir, search_dir_name)
    raise NotFoundError('Error')


def split_annotation(annotation_file_path, annotation_file_format, classes_file_path, n_splits=5):
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

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    idx = 0
    for train_annotation_line_idx_list, test_annotation_line_idx_list in cv.split(annotation_lines):
        print(f"{len(train_annotation_line_idx_list)} train annotations and {len(test_annotation_line_idx_list)} test annotations")

        train_annotation_lines = [annotation_lines[train_annotation_line_idx] for train_annotation_line_idx in train_annotation_line_idx_list]
        test_annotation_lines = [annotation_lines[test_annotation_line_idx] for test_annotation_line_idx in test_annotation_line_idx_list]

        split_dir = os.path.join(output_dir, '%03d' % idx)
        os.makedirs(split_dir, exist_ok=True)

        shutil.copyfile(classes_file_path, os.path.join(split_dir, "classes.txt"))

        with open(os.path.join(split_dir, "train.txt"), "w") as f:
            train_annotation_output = "\n".join(train_annotation_lines)
            f.write(train_annotation_output)

        with open(os.path.join(split_dir, "test.txt"), "w") as f:
            test_annotation_output = "\n".join(test_annotation_lines)
            f.write(test_annotation_output)

        print(f"Saved train.txt and test.txt in {split_dir}")
        if annotation_file_format != 'bbox':
            return

        ground_truth_output_dir = os.path.join(split_dir, "ground-truth")
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
        idx += 1


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
        '-s', '--n_splits', type=int,
        help='Number of folds. Must be at least 2.',
        default=5
    )

    args = vars(parser.parse_args())

    if (args["n_splits"] < 2):
        sys.exit("n_sprits must be greater than or equal to 2.")

    split_annotation(annotation_file_path=args["annotation_file_path"], annotation_file_format=args['annotation_file_format'], classes_file_path=args["classes_file_path"], n_splits=args["n_splits"])

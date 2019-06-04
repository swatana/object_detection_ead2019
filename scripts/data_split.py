import os
import argparse
from sklearn.model_selection import train_test_split
from random import shuffle
from collections import defaultdict
import shutil


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
        while idx < len(train_items) and len(test_items) >= test_ratio * len(items):
            test_items.append(train_items[idx])
            idx += 1

    return train_items[idx:], test_items


def split_annotation(annotation_file, test_classes_path, lower_bound=None, test_ratio=0.2): # "data_labels/coco_fire"
    annotation_dir = os.path.dirname(annotation_file)
    dataset_name = os.path.basename(annotation_dir)
    upper_annnotation_dir = os.path.dirname(annotation_dir)
    if upper_annnotation_dir == '':
        upper_annnotation_dir = '.'
    output_dir = get_unused_dir_num(upper_annnotation_dir, dataset_name)
    with open(test_classes_path) as fp:
        test_classes = [line.strip() for line in fp]
    with open(annotation_file) as f:
        annotation_list = [line for line in f]

    os.makedirs(output_dir, exist_ok=True)

    shutil.copyfile(test_classes_path, os.path.join(output_dir, "classes.txt"))

    if lower_bound:
        objects_in_file = dict()
        all_string = dict()
        train_items = []
        for item in annotation_list:
            filename, *objects = item.split()
            all_string[filename] = item
            objects_in_file[filename] = objects
            class_ids = [int(obj.split(',')[-1]) for obj in objects]
            train_items.append((filename, class_ids))

        train_img_path, test_img_path = select_sample(train_items, lower_bound, test_ratio)

        make_annotation = lambda filename : all_string[filename]
        train_annotations = list(map(make_annotation, train_img_path))
        test_annotations = list(map(make_annotation, test_img_path))
    else:
        train_annotations, test_annotations = train_test_split(
            annotation_list, test_size=test_ratio, random_state=42)

    print("{num_train} train annotations and {num_test} test annotations".format(num_test=len(test_annotations), num_train=len(train_annotations)))
    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        for line in train_annotations:
            print(line, end="", file=f)

    with open(os.path.join(output_dir, "test.txt"), "w") as f:
        for line in test_annotations:
            print(line, end="", file=f)

            gt_output_dir = os.path.join(output_dir, "ground-truth")
            os.makedirs(gt_output_dir, exist_ok=True)

            line = line.split()
            img_path = line[0]
            bbox = line[1:]

            img_basename = os.path.basename(img_path)
            img_basename, _ = os.path.splitext(img_basename)

            with open(os.path.join(gt_output_dir, img_basename + ".txt"), "w") as fg:
                # pprint(gt_list)
                for b in bbox:
                    obj = b.split(",")
                    x_min = obj[0]
                    y_min = obj[1]
                    x_max = obj[2]
                    y_max = obj[3]
                    class_id = int(obj[4])
                    print(class_id)
                    print(test_classes[class_id])
                    print(
                        "{class_name}\t{coordinates}".format(
                            class_name=test_classes[class_id],
                            coordinates="{0}\t{1}\t{2}\t{3}".format(
                                x_min, y_min, x_max, y_max),
                        ),
                        end="\n",
                        file=fg
                    )

    print("Saved train.txt and test.txt in {}".format(output_dir))
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--annotation_file', type=str, default=None, required=True,
        help='path to annotation file'
    )
    
    parser.add_argument(
        '-c', '--classes', type=str,
        help='path to class definitions'
    )

    parser.add_argument(
        '-r', '--test_ratio', type=float,
        help='ratio of test annotations'
    )

    parser.add_argument(
        "-n",
        "--lower_bound",
        help="The least number of apperance of each class",
        type=int
    )

    args = vars(parser.parse_args())

    split_annotation(annotation_file=args["annotation_file"], test_classes_path=args["classes"], lower_bound=args["lower_bound"], test_ratio=args["test_ratio"])
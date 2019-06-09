"""
Mask R-CNN
Train on your own dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------
Usage: import the module or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 own.py train -a=/path/to/your/own/annotation/file.txt

    # Resume training a model that you had trained earlier
    python3 own.py train -a=/path/to/your/own/annotation/file.txt -w=last

    # Train a new model starting from ImageNet weights
    python3 own.py train -a=/path/to/your/own/annotation/file.txt -w=imagenet

    # Apply color splash to an image
    python3 own.py splash -w=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 own.py splash -w=last -v=<URL or path to file>
"""

from pprint import pprint
import os
import sys
import json
import datetime
import numpy as np
import shutil
import skimage.draw
import re
from sklearn.model_selection import train_test_split
import ast

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils

import imgaug

# Path to trained weights file
COCO_WEIGHTS_PATH = "mask_rcnn_coco.h5"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "logs"

############################################################
#  Configurations
############################################################


class OwnConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    def __init__(self, dataset_dir, args):
        # Give the configuration a recognizable name
        self.NAME = os.path.basename(dataset_dir)
        self.LAYERS = args.layers

        # Number of classes (including background)
        self.NUM_CLASSES = sum(1 for line in open(os.path.join(dataset_dir, "classes.txt")))
        super().__init__()

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class OwnConfig_new(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    def __init__(self, classes_path):
        # Give the configuration a recognizable name
        self.NAME = os.path.basename(os.path.dirname(classes_path))

        # Number of classes (including background)
        self.NUM_CLASSES = sum(1 for line in open(classes_path))
        super().__init__()

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

class OwnDataset(utils.Dataset):

    def load_own(self, dataset_dir, annotations):
        """Load a subset of your own dataset.
        dataset_dir: Root directory of the dataset.
        """
        dataset_name = os.path.basename(dataset_dir)
        class_names = get_classes(os.path.join(dataset_dir, "classes.txt"))
        num_classes = len(class_names)

        assert class_names[0] == "BG", "class.txt must contain BG in the first line"
        for class_name in class_names[1:]:
            self.add_class(dataset_name, len(self.class_info), class_name)

        for annotation in annotations:
            # print(annotation)
            polygons = []
            local_class_ids = []
            if "{" in annotation:
                image_path, polygons = annotation.split(None, 1)
                polygons = json.loads(polygons)
                for i, polygon in enumerate(polygons):
                    polygons[i]['name'] = 'polygon'
                    local_class_ids.append(polygon['class_id'])
            else:
                image_path, *masks = annotation.split()
                for mask in masks:
                    if "[" in mask:
                        mask = ast.literal_eval(mask)
                        areas = mask[0]
                        local_class_ids.append(int(mask[-1]))
                        polygons.append({'all_points_x': [[int(x) for x in area[::2]] for area in areas], 'all_points_y': [[int(y) for y in area[1::2]] for area in areas], 'name': 'polygon'})
                    else:
                        mask = mask.split(",")
                        local_class_ids.append(int(mask[-1]))
                        polygons.append({'all_points_x': [[int(x) for x in mask[:-1:2]]], 'all_points_y': [[int(y) for y in mask[1::2]]], 'name': 'polygon'})
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            # print(mage_path, height, width)
            self.add_image(
                dataset_name,
                image_id=annotation[0],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                local_class_ids=np.array(local_class_ids, dtype=np.int32)
            )


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, areas in enumerate(info["polygons"]):
            for (xs, ys) in zip(areas['all_points_x'], areas['all_points_y']):
                # Get indexes of pixels inside the polygon and set them to 1
                rr, cc = skimage.draw.polygon(ys, xs)
                mask[rr, cc, i] ^= 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), info["local_class_ids"]

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]



def train(model, annotations_path, test_run=False):
    """Train the model."""
    with open(annotations_path) as f:
        annotations = f.readlines()

    if test_run:
        annotations = annotations[:min(len(annotations), 200)]

    dataset_dir = os.path.dirname(annotations_path)
    train_annotations, val_annotatinons = train_test_split(
            annotations, test_size=0.1, random_state=42)

    # Training dataset.
    dataset_train = OwnDataset()
    print(dataset_dir)
    dataset_train.load_own(dataset_dir, train_annotations)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = OwnDataset()
    dataset_val.load_own(dataset_dir, val_annotatinons)
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.

    augmentation = imgaug.augmenters.Sometimes(0.5, [
        imgaug.augmenters.Fliplr(0.5),
        imgaug.augmenters.Flipud(0.5),
        imgaug.augmenters.GaussianBlur(sigma=(0.0, 1.0))
    ])

    if not os.path.exists(model.log_dir):
        os.makedirs(model.log_dir)
    shutil.copyfile(os.path.join(dataset_dir, "classes.txt"), os.path.join(model.log_dir ,"classes.txt"))
    with open(os.path.join(model.log_dir ,"config.txt"), "w") as f:
        """Display Configuration values."""
        print("\nConfigurations:", file=f)
        for a in dir(config):
            if not a.startswith("__") and not callable(getattr(config, a)):
                print("{:30} {}".format(a, getattr(config, a)), file=f)
        print("\n", file=f)

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers=config.LAYERS,
                augmentation=False)

    shutil.copyfile(os.path.join(dataset_dir, "classes.txt"), os.path.join(model.log_dir ,"classes.txt"))




############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('-a', '--annotations', required=False,
                        metavar="/path/to/annotations.txt",
                        help='dataset')
    parser.add_argument('-w', '--weights', required=False,
                        default='coco',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('-l', '--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('-ly', '--layers', required=False,
                        default="heads",
                        help='Layers trainable')

    parser.add_argument('-i', '--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('-v', '--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.annotations, "Argument --annotations is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    dataset_dir = os.path.dirname(args.annotations)
    print("Weights: ", args.weights)
    print("Dataset: ", dataset_dir)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = OwnConfig(dataset_dir, args)
    else:
        class InferenceConfig(OwnConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig(dataset_dir)
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=args.logs)
    # model = []
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    train(model, args.annotations, test_run=False)

import sys
import argparse
import glob
import os
from PIL import Image

class NotFoundError(Exception):
    pass

def get_unused_dir_num(pdir, pref=None):
    os.makedirs(pdir, exist_ok=True)
    dir_list = os.listdir(pdir)
    for i in range(1000):
        search_dir_name = "" if pref is None else (
            pref + "_" ) + '%03d' % i
        if search_dir_name not in dir_list:
            return os.path.join(pdir, search_dir_name)
    raise NotFoundError('Error')

def detect_img(model):

    image_glob = FLAGS.image_glob
    test_file = FLAGS.test_file
    print(image_glob)
    print(FLAGS.model_path)
    result_name = os.path.basename(FLAGS.model_path)

    image_source = ""
    if image_glob:
        img_path_list = glob.glob(image_glob)
        image_source = image_glob
    else:
        with open(test_file) as f:
            img_path_list = [line.strip().split()[0] for line in f]
        image_source = test_file

    pdir = os.path.join(
        "results", FLAGS.network, os.path.basename(
            os.path.dirname(image_source)))

    output_dir = get_unused_dir_num(pdir=pdir, pref=result_name)
    image_output_dir = os.path.join(output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)
    prediction_output_dir = os.path.join(
        output_dir, "predictions")
    os.makedirs(prediction_output_dir, exist_ok=True)

    for img_path in img_path_list:
        img_basename, _ = os.path.splitext(os.path.basename(img_path))
        try:
            image = Image.open(img_path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, objects = model.detect_image(image)
            r_image.save(
                os.path.join(
                    image_output_dir,
                    img_basename + ".jpg",
                ))

            with open(
                    os.path.join(
                        prediction_output_dir, img_basename + ".txt"
                    ),
                    "w") as f:
                for obj in objects:
                    class_name = obj["class"]
                    score = obj["score"]
                    x_min, y_min, x_max, y_max = obj["bbox"]
                    print(
                        "{class_name}\t{score}\t{coordinates}".format(
                            score=score,
                            class_name=class_name,
                            coordinates="{0}\t{1}\t{2}\t{3}".format(
                                x_min, y_min, x_max, y_max),
                        ),
                        end="\n",
                        file=f
                    )

    model.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '-m', '--model_path', type=str,
        help='path to model weight file'
    )

    parser.add_argument(
        '-a', '--anchors_path', type=str,
        help='path to anchor definitions'
    )

    parser.add_argument(
        "-c", '--classes_path', type=str,
        help='path to class definitions'
    )

    parser.add_argument(
        "-i", "--image_glob", nargs='?', type=str, default=None,
        help="Image glob pattern"
    )
    parser.add_argument(
        "-t", "--test_file", nargs='?', type=str, default=None,
        help="test file path"
    )
    parser.add_argument(
        '-n',
        '--network',
        type=str,
        choices=[
            'yolo',
            'mrcnn',
            'keras-centernet'],
        default='yolo',
        help='Network structure')

    FLAGS = parser.parse_args()

    if FLAGS.network == "yolo":
        from yolo import YOLO, detect_video
        model = YOLO(**vars(FLAGS))

    elif FLAGS.network == "mrcnn":
        from mask_rcnn import MaskRCNN
        model = MaskRCNN(FLAGS.model_path, FLAGS.classes_path)

    elif FLAGS.network == "keras-centernet":
        from centernet import CENTERNET
        model = CENTERNET(FLAGS.model_path, FLAGS.classes_path)

    else:
        parser.error("Unknown network")
    detect_img(model)


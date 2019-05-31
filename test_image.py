import sys
import argparse
import glob
import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2
import colorsys

class NotFoundError(Exception):
    pass

def make_r_image(image, objects, colors, alpha=0.3):
    if len(objects) and 'mask' in objects[0]:
        image = np.array(image)
        image = np.array(image[..., ::-1], dtype=np.float32)
        polygons = []
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            score = obj['score']
            class_name = obj['class_name']
            class_id = obj['class_id']
            mask = obj['mask']
            color = colors[class_id]

            image[mask] = (image[mask] * (1.0 - alpha) + np.array(color) * alpha)

            # Draw mask boundaries
            contours = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[-2]

            polygon = []

            for cnt in contours:
                ps=[]
                if (len(cnt) <= 2):
                    continue
                for point in cnt:
                    x, y = point[0]
                    ps.append(int(x))
                    ps.append(int(y))
                polygon.append(ps)

            polygons.append(polygon)

            cv2.drawContours(image, contours, -1, color, 2)

            # Draw rectangular bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Label
            cv2.putText(image, "%i %s %.2f" % (class_id, class_name, score),
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 255, 255))

        image = image[..., ::-1]  # Convert back to PIL
        return Image.fromarray(image.astype(np.uint8))
    else:
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for obj in reversed(objects):
            left, top, right, bottom = obj['bbox']
            score = obj['score']
            class_name = obj['class_name']
            classs_id = obj['class_id']
            color = colors[classs_id]

            label = '{} {:.2f}'.format(class_name, score)
            draw = ImageDraw.Draw(image)

            label_size = draw.textsize(label, font)
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=color)
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=color)
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        return image

def deprocess_feature(x):
    """Deprocess feature x for visualization

    Args:
    - x: 3 dimensional vector (w x h x depth)
    """

    x = x - np.mean(x, axis=-1, keepdims=True)
    x /= np.std(x, axis=-1, keepdims=True) + 1e-5
    return np.clip(x * 0.1 + 0.5, 0.0, 1.0)

def visualize_and_save(feature, save_path):
    """Visualize feature values"""

    feature = deprocess_feature(feature)
    fig, axes = plt.subplots(6, 10, sharex=True, sharey=True)
    for index, axis in enumerate(axes.flat):
        plotted_im = axis.imshow(feature[..., index])
        axis.set_axis_off()
    fig.colorbar(plotted_im, ax=axes.ravel().tolist())
    plt.savefig(save_path)

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
    feature_output_dir = os.path.join(output_dir, "feature")
    os.makedirs(feature_output_dir, exist_ok=True)

    # Generate colors for drawing bounding boxes.
    class_num = 0
    classes_path = os.path.expanduser(FLAGS.classes_path)
    with open(classes_path) as f:
        class_num = len(f.readlines())
    hsv_tuples = [(x / class_num, 1., 1.)
                    for x in range(class_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    for img_path in img_path_list:
        img_basename, _ = os.path.splitext(os.path.basename(img_path))
        try:
            image = Image.open(img_path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            result = model.detect_image(image)
            # r_image = result['r_image']
            objects = result['objects']
            r_image = make_r_image(image, objects, colors)
            r_image.save(
                os.path.join(
                    image_output_dir,
                    img_basename + ".jpg",
                ))

            if 'feature' in result:
                feature = result['feature']
                visualize_and_save(feature, os.path.join(feature_output_dir, img_basename + ".png"))
                np.save(
                    os.path.join(
                        feature_output_dir,
                        img_basename +
                        ".npy"),
                    feature)
            with open(
                    os.path.join(
                        prediction_output_dir, img_basename + ".txt"
                    ),
                    "w") as f:
                for obj in objects:
                    class_name = obj["class_name"]
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
        from yolo import YOLO
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


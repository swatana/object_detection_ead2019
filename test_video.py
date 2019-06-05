import sys
import argparse
from PIL import Image
from timeit import default_timer as timer
import numpy as np
import test_image
import colorsys
import os
from utils import generate_colors, make_r_image
 
def detect_img(model):

    class_num = 0
    classes_path = os.path.expanduser(FLAGS.classes_path)
    with open(classes_path) as f:
        class_num = len(f.readlines())
    colors = generate_colors(class_num)

    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            result = model.detect_image(image)
            objects = result['objects']
            r_image = test_image.make_r_image(image, objects, colors)
            r_image.show()
    model.close_session()


def detect_video(model, video_path, output_path=""):
    import cv2

    # Generate colors for drawing bounding boxes.
    class_num = 0
    classes_path = os.path.expanduser(FLAGS.classes_path)
    with open(classes_path) as f:
        class_num = len(f.readlines())
    colors = generate_colors(class_num)

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

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        result = model.detect_image(image)
        objects = result['objects']
        r_image = make_r_image(image, objects, colors)
        result = np.asarray(r_image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    model.close_session()


FLAGS = None

# In[2]:
if __name__ == '__main__':
# In[3]:
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''

    parser.add_argument(
        '-f', '--file', type=str,
        help='file'
    )

    parser.add_argument(
        '-m', '--model', type=str,
        help='path to model weight file'
    )

    parser.add_argument(
        '-a', '--anchors', type=str,
        help='path to anchor definitions'
    )

    parser.add_argument(
        "-c", '--classes_path', type=str,
        default="model_data/yolo3/coco/classes.txt",
        help='path to class definitions'
    )

    parser.add_argument(
        "-g", '--gpu_num', type=int, default=1,
        help='Number of GPU to use'
    )
    
    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default=0,
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
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
# In[4]:
    

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        detect_img(model)
    elif "input" in FLAGS:
        detect_video(model, FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")

#%%

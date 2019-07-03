import sys
import argparse
import glob
from yolo import YOLO, detect_video
from PIL import Image

count = 0

def person_exist_in_obj(obj):
    i = 0
    while True:
        try:
            if obj[i]["class"] == "person":
                return True
            i += 1
        except:
            break
    return False

def crop_person(image, obj):
    global count
    i = 0
    while True:
        try:
            if obj[i]["class"] == "person":
                fname = './crop_person_images/' + str(count) + '.jpg'
                img_crop = image.crop(obj[i]["bbox"])
                img_crop.save(fname)
                count += 1
                # img_crop.show()
            i += 1
        except:
            break
    return False


def detect_img(yolo):
    image_path = glob.glob('./mscoco2017/train2017/*.jpg')
    i = 0
    while True:
        try:
            img = image_path[i]
            print(img)
            i += 1
        except:
            break
        try:
            image = Image.open(img)
            original_image = image.copy()
        except:
            print('Open Error! Try again!')
            continue
        else:
            try:
                r_image, obj = yolo.detect_image(image)
            except:
                continue
            if person_exist_in_obj(obj):
                crop_person(original_image, obj)

            # r_image.show()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''

    parser.add_argument(
        '-m', '--model', type=str,
        help='path to model weight file'
    )

    parser.add_argument(
        '-a', '--anchors', type=str,
        help='path to anchor definitions'
    )

    parser.add_argument(
        "-c", '--classes', type=str,
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

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")

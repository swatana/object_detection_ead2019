import cv2
import os
from argparse import ArgumentParser

def get_image_name(output_dir):
    global n
    file_list = os.listdir(output_dir)
    for i in range(n, 100000):
        search_filename = '%07d.jpg' % i
        if search_filename not in file_list:
            n = i+1
            return output_dir + search_filename 
    raise NotFoundError('Error')

def main(args):
    cap = cv2.VideoCapture(args.video_name)   
    output_dir = './images/' + args.category + '/'
    image_count = 0
    step = args.step

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if image_count % step == 0:
            image_name = get_image_name(output_dir)
            cv2.imwrite(image_name, frame)
        image_count += 1
    cap.release()

n = 0
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-v', '--video_name', type=str, default='video.avi', help = 'file name of video')
    parser.add_argument('-c', '--category', type=str, default='others', help = 'category of images')
    parser.add_argument('-s', '--step',type=int, default=1, help = 'interval of frame')
    main(parser.parse_args())

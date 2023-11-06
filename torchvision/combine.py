import argparse
import os
import glob
import datetime
import shutil

yolo_classes = ["artifact", "contrast", "instrument", "saturation"]
faster_rcnn_classes = ["blur", "bubbles", "specularity"]

def conbine_result():
    yolo_file_path = FLAGS.yolo_result_dir_path
    faster_rcnn_file_path = FLAGS.faster_rcnn_result_dir_path
    predictions_dict = {}
    date=datetime.datetime.now().strftime('%m%d%H%M')
    if not os.path.exists(f'./results/combined/{date}/predictions'):  
        os.makedirs(f'./results/combined/{date}/predictions')
    shutil.copyfile(os.path.join(yolo_file_path,"classes.txt"), f'./results/combined/{date}/classes.txt')
    shutil.copyfile(os.path.join(yolo_file_path,"test.txt"), f'./results/combined/{date}/test.txt')
    for file_path in glob.glob(os.path.join(yolo_file_path, 'predictions/*.txt')):
        output=""
        image_name = os.path.splitext(os.path.basename(file_path))[0]
        predictions_dict[image_name] = []
        with open(file_path, 'r') as f:
            for line in f:
                class_name, score, sx, sy, ex, ey = line.split('\t')
                if class_name in yolo_classes:
                    output += f"{class_name}\t{score}\t{sx}\t{sy}\t{ex}\t{ey}"
        
        with open(os.path.join(faster_rcnn_file_path,"predictions",image_name+".txt"), 'r') as f:
            for line in f:
                class_name, score, sx, sy, ex, ey = line.split('\t')
                if class_name in faster_rcnn_classes:
                    output += f"{class_name}\t{score}\t{sx}\t{sy}\t{ex}\t{ey}"

        with open(f"./results/combined/{date}/predictions/{image_name}.txt","w") as f:
            print(output, end="", file=f)
   
    


FLAGS=None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-yr', '--yolo_result_dir_path', type=str, required=True,
        help='path to the first result directory path'
    )
    parser.add_argument(
        '-fr', '--faster_rcnn_result_dir_path', type=str, required=True,
        help='path to the first result directory path'
    )
    FLAGS = parser.parse_args()

    conbine_result()

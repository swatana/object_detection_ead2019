import argparse
import cv2
import glob
import os
import numpy as np

def ead2019_object_detection_data_to_annotation(ead2019_object_detection_data_input_dir_path, output_dir_path, output_format):
    ead2019_object_detection_data_input_txt_file_names = glob.glob(os.path.join(ead2019_object_detection_data_input_dir_path, "*.txt"))

    output_file_name = os.path.join(output_dir_path, "annotation.txt")

    output_rows = []
    for ead2019_object_detection_data_input_txt_file_name in ead2019_object_detection_data_input_txt_file_names:
        output_row_elements = []
        ead2019_object_detection_data_input_jpg_file_name = os.path.splitext(ead2019_object_detection_data_input_txt_file_name)[0] + ".jpg"
        output_row_elements.append(ead2019_object_detection_data_input_jpg_file_name)

        ead2019_object_detection_data_input_jpg = cv2.imread(ead2019_object_detection_data_input_jpg_file_name)[:,:,::-1]
        ead2019_object_detection_data_input_jpg_height, ead2019_object_detection_data_input_jpg_width, _ = ead2019_object_detection_data_input_jpg.shape

        with open(ead2019_object_detection_data_input_txt_file_name) as f:
            ead2019_object_detection_data_input_rows = [line.strip() for line in f]

        for ead2019_object_detection_data_input_row in ead2019_object_detection_data_input_rows:
            ead2019_object_detection_data_input_row_split = ead2019_object_detection_data_input_row.split()

            bbox_class = ead2019_object_detection_data_input_row_split[0]
            bbox_center_rx = float(ead2019_object_detection_data_input_row_split[1])
            bbox_center_ry = float(ead2019_object_detection_data_input_row_split[2])
            bbox_rwidth = float(ead2019_object_detection_data_input_row_split[3])
            bbox_rheight = float(ead2019_object_detection_data_input_row_split[4])

            bbox_x1 = (bbox_center_rx - bbox_rwidth / 2.) * ead2019_object_detection_data_input_jpg_width
            bbox_y1 = (bbox_center_ry - bbox_rheight / 2.) * ead2019_object_detection_data_input_jpg_height
            bbox_x2 = (bbox_center_rx + bbox_rwidth / 2.) * ead2019_object_detection_data_input_jpg_width
            bbox_y2 = (bbox_center_ry + bbox_rheight / 2.) * ead2019_object_detection_data_input_jpg_height

            bbox_x1 = int(np.clip(bbox_x1, 0, ead2019_object_detection_data_input_jpg_width - 1))
            bbox_y1 = int(np.clip(bbox_y1, 0, ead2019_object_detection_data_input_jpg_height - 1))
            bbox_x2 = int(np.clip(bbox_x2, 0, ead2019_object_detection_data_input_jpg_width - 1))
            bbox_y2 = int(np.clip(bbox_y2, 0, ead2019_object_detection_data_input_jpg_height - 1))

            bbox_x1 = str(bbox_x1)
            bbox_y1 = str(bbox_y1)
            bbox_x2 = str(bbox_x2)
            bbox_y2 = str(bbox_y2)

            if output_format == "yolo":
                output_bbox_comma = f'{bbox_x1},{bbox_y1},{bbox_x2},{bbox_y2},{bbox_class}'
                output_row_elements.append(output_bbox_comma)
            elif output_format == "mrcnn":
                output_mask = f'[[[{bbox_x1},{bbox_y1},{bbox_x1},{bbox_y2},{bbox_x2},{bbox_y2},{bbox_x2},{bbox_y1}]],{bbox_class}]'
                output_row_elements.append(output_mask)
        output_row = " ".join(output_row_elements)
        output_rows.append(output_row)

    output_str = "\n".join(output_rows)

    with open(output_file_name, 'w') as f:
        f.write(output_str)

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--ead2019_object_detection_data_input_dir_path', help='path to input dir of ead2019(object detection)', required=True)
    parser.add_argument('-o', '--output_dir_path', help='path to output dir', required=True)
    parser.add_argument('-f', '--format', help='output format of annotation file', required=False, default='yolo', choices=['yolo', 'mrcnn'])
    args = vars(parser.parse_args())

    ead2019_object_detection_data_to_annotation(args["ead2019_object_detection_data_input_dir_path"], args["output_dir_path"], args["format"])


if __name__ == '__main__':
    _main()

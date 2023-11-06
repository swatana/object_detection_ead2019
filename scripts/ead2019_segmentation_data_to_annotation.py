import argparse
import glob
import os

def ead2019_segmentation_data_to_annotation(input_image_dir_path, input_mask_dir_path, output_dir_path):
    input_image_file_path_list = glob.glob(os.path.join(input_image_dir_path, "*.jpg"))

    input_mask_file_path_list = glob.glob(os.path.join(input_mask_dir_path, "*.tif"))

    input_file_basename_without_ext_2_input_mask_file_path = dict()
    for input_mask_file_path in input_mask_file_path_list:
        input_mask_file_basename = os.path.basename(input_mask_file_path)
        input_mask_basename_without_ext, _ = os.path.splitext(input_mask_file_basename)

        input_file_basename_without_ext_2_input_mask_file_path[input_mask_basename_without_ext] = input_mask_file_path

    output_file_name = os.path.join(output_dir_path, "annotation.txt")

    output_annotation_rows = []
    for input_image_file_path in input_image_file_path_list:
        input_image_file_basename = os.path.basename(input_image_file_path)
        input_image_file_basename_without_ext, _ = os.path.splitext(input_image_file_basename)

        input_mask_file_path = input_file_basename_without_ext_2_input_mask_file_path.get(input_image_file_basename_without_ext)

        if input_mask_file_path is None:
            continue

        output_annotation_row = f"{input_image_file_path} {input_mask_file_path}"
        output_annotation_rows.append(output_annotation_row)

    output_annotation = "\n".join(output_annotation_rows)

    with open(output_file_name, 'w') as f:
        f.write(output_annotation)

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ead2019_segmentation_data_input_image_dir_path', help='path to input image dir of ead2019(segmentation)', required=True)
    parser.add_argument('-m', '--ead2019_segmentation_data_input_mask_dir_path', help='path to input mask dir of ead2019(segmentation)', required=True)
    parser.add_argument('-o', '--output_dir_path', help='path to output dir', required=True)
    args = vars(parser.parse_args())

    ead2019_segmentation_data_to_annotation(args["ead2019_segmentation_data_input_image_dir_path"], args["ead2019_segmentation_data_input_mask_dir_path"], args["output_dir_path"])


if __name__ == '__main__':
    _main()

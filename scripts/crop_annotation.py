import argparse
import os
import re
from PIL import Image

def crop_annotation(annotation_path):
    root_dir = os.path.dirname(annotation_path)
    new_lines = []

    with open(annotation_path) as f:
        annotations = [line.strip().split() for line in f]
    with open(os.path.join(root_dir, "classes.txt")) as f:
        class_names = ["_".join(line.strip().split()) for line in f]

    for annotation in annotations[:100]:
        cropped_image_dir = os.path.join(os.path.join(root_dir, "cropped_images"), os.path.splitext(os.path.basename(annotation[0]))[0])
        os.makedirs(cropped_image_dir, exist_ok=True)
        img = Image.open(annotation[0])

        for areas in annotation[1:]:
            dots = [int(x) for x in re.split(r'[\[\]\,]', areas) if x != ""]
            xs = dots[:-1:2]
            ys = dots[1::2]
            cat = class_names[dots[-1]]
            crop = (max(0, min(xs) - 1), max(0, min(ys) - 1), min(img.size[0], max(xs) + 1), min(img.size[1], max(ys) + 1), cat)
            image_id = os.path.basename(annotation[0])
            image_id, _ = os.path.splitext(image_id)
            image_base_name = image_id + "_" + "_".join([str(s) for s in crop])
            print("image_base_name",image_base_name)
            cropped_image_path = os.path.join(cropped_image_dir, image_base_name) + ".jpg"
            img.crop(crop[:4]).save(cropped_image_path)
            print("Saved " + cropped_image_path)

            areas = [x for x in re.split(r'[\[\]]', areas) if x != ""]
            if len(dots) == 5:
                areas = [area.split(",") for area in areas]
                new_line = cropped_image_path + " "
                for area in areas:
                    new_line += ",".join([str(int(x) - crop[i%2]) for i, x in enumerate(area[:-1])]) 
                new_line += "," + str(areas[-1][-1])
                new_lines.append(new_line)
            else:
                new_line = cropped_image_path + " [[" + ",".join([str(int(x) - crop[i%2]) for i, x in enumerate(dots[:-1])]) + "]]," + str(dots[-1])
                new_lines.append(new_line)

    if len(new_lines) > 0:
        cropped_annotation_path = os.path.join(root_dir, "cropped_annotation.txt")
        with open(cropped_annotation_path, "w") as f:
            for new_line in new_lines:
                print(new_line, file=f)
        print("Saved " + cropped_annotation_path)



if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '-a', '--annotation_path', type=str, default=None, required=True,
        help='path to annotation file'
    )
    crop_annotation(ap.parse_args().annotation_path)

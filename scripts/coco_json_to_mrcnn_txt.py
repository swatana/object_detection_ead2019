
# In[0]:
import json
from collections import defaultdict
from pprint import pprint
import numpy as np

# In[1]:
img_dir = "mscoco2017/train2017/"
annotation_file = "mscoco2017/annotations/instances_train2017.json"
dataset = json.load(open(annotation_file, 'r'))
dataset = dataset

# In[2]:
print('creating index...')
anns, cats, imgs = {}, {}, {}
imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
if 'annotations' in dataset:
    for ann in dataset['annotations']:
        imgToAnns[ann['image_id']].append(ann)
        anns[ann['id']] = ann

if 'images' in dataset:
    for img in dataset['images']:
        imgs[img['id']] = img

if 'categories' in dataset:
    for cat in dataset['categories']:
        cats[cat['id']] = cat

if 'annotations' in dataset and 'categories' in dataset:
    for ann in dataset['annotations']:
        catToImgs[ann['category_id']].append(ann['image_id'])

print('index created!')

# In[3]:

pprint(cats)
class_txt = "BG\n"
for cat in cats.values():
    # print(cat)
    class_txt += "{}\n".format(cat['name'].replace(' ', '_'))

# print(class_txt)
with open("data_labels/polygon/coco/classes.txt", mode='w') as f:
    f.write(class_txt)

# In[4]:
print("In[4]")
list_line = ""
json_line = ""
for img_id in imgToAnns:
    # pprint(imgs[img_id])
    new_list_line = ""
    new_json_line = ""
    jsons = []
    # print (new_list_line)
    for imgToAnn in imgToAnns[img_id]:
        # pprint(imgToAnn)
        # pprint(imgToAnn['segmentation'])
    
        if isinstance(imgToAnn['segmentation'], dict):
            # print(imgToAnn['segmentation'])
            continue
        segmentation = ""
        # print(imgToAnn['segmentation'])
        # polygon = map(lambda x: str(int(x)), imgToAnn['segmentation'])
        # areas = np.array(imgToAnn['segmentation']).astype(int).tolist()

        areas = [[int(x) for x in area[::]] for area in imgToAnn['segmentation']]
 
        # print(polygon)
        cat = imgToAnn['category_id']
        if cat >= 13 and cat <= 25:
            cat -= 1
        elif cat >= 27 and cat <= 28:
            cat -= 2
        elif cat >= 31 and cat <= 44:
            cat -= 4
        elif cat >= 46 and cat <= 65:
            cat -= 5
        elif cat == 67:
            cat -= 6
        elif cat == 70:
            cat -= 8
        elif cat >= 72 and cat <= 82:
            cat -= 9
        elif cat >= 84 and cat <= 90:
            cat -= 10
        # if cat != 1:
        #     continue
        new_list_line += " "+"[{},{}]".format(areas, cat).replace(' ', '')
        jsons.append(
            {'all_points_x': [[int(x) for x in area[::2]] for area in areas], 
            'all_points_y': [[int(y) for y in area[1::2]] for area in areas], 
            "class_id": cat})

    if new_list_line != "":
        list_line += "{}{}".format(img_dir, imgs[img_id]['file_name']) + new_list_line + "\n"
        json_line += "{}{} ".format(img_dir, imgs[img_id]['file_name']) + str(jsons) + "\n"
#     break
# print (list_line)
# print (json_line) 
# In[5]:
with open("data_labels/polygon/coco/train_list.txt", mode='w') as f:
    f.write(list_line)
with open("data_labels/polygon/coco/train_json.txt", mode='w') as f:
    f.write(json_line)

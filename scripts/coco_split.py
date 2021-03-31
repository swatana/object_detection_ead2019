import cv2

import glob
import json

fdir = 'argus.orig/'
tdir = 'argus.100/'

files = glob.glob(fdir+"/*.json")
files.sort()


print(files)
num = 10
for f in files:
    t = f.replace("argus.orig", "argus100")
    
    with open(f) as json_file:
        data = json.load(json_file)
        print(data)
        img_name=data['images'][0]['file_name']
        height=data['images'][0]['height']
        width=data['images'][0]['width']

        w = width//num
        h = height//num
        for i in range(0,width,w):
            for j in range(0,height,h):
                js = {'images':[],'categories':[],'annotations':[]}
                js['images'].append(data['images'][0])
                js['images'][0]['file_name'] = img_name.replace(".jpg", f"-{i}-{j}.jpg")
                js['categories'] = data['categories']
                print(i,i+w,j,j+h)
                tl = t.replace(".json", f"-{i}-{j}.json")
                for a in data['annotations']:
                    if i < a['bbox'][0] and a['bbox'][0] < (i+w) and j < a['bbox'][1] and a['bbox'][1] < (j+h): 
                        a['bbox'][0] -= i
                        a['bbox'][1] -= j
                        # for s in a['segmentation'][0]:
                        a['segmentation'][0][::2] = list(map(lambda x: x - i, a['segmentation'][0][::2]))
                        a['segmentation'][0][1::2] = list(map(lambda x: x - j, a['segmentation'][0][1::2]))
                        js['annotations'].append(a)
                with open(tl, 'w') as outfile:
                    json.dump(js, outfile)
    # break


files = glob.glob("argusall/*.jpg")
files.sort()

# print(files)
num = 10
for f in files:
    
    img_name=f

    res_name = img_name.replace("argusall", "argusall100")
    print(res_name)

    img = cv2.imread(img_name)
    height, width, channels = img.shape
    h = height//num
    w = width//num
    for i in range(0,width,w):
        for j in range(0,height,h):
            print(i,i+w,j,j+h)
            clp = img[j:j+h, i:i+w]
            tl = res_name.replace(".jpg", f"-{i}-{j}.jpg")
            cv2.imwrite(tl, clp)   
    # break


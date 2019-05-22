# object-detection

## How to Use

## Prepare package
[InstallCUDA](/InstallCUDA.md)
```
git clone https://github.com/qqwweee/keras-yolo3
ln -s keras-yolo3/yolo3 .
pip3 install -r requirements.txt

git clone https://github.com/matterport/Mask_RCNN.git
ln -s Mask_RCNN/mrcnn .
pip3 install -r Mask_RCNN/requirements.txt
```

## Prepare Model

```
wget https://pjreddie.com/media/files/yolov3.weights
python3 keras-yolo3/convert.py yolov3.cfg yolov3.weights model_data/yolo3/coco/yolo.h5
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 -P model_data/mrcnn/coco/
```

## Run

```
python3 test_video.py --model=model_data/yolo3/coco/yolo.h5 --anchors=model_data/yolo3/coco/yolo_anchors.txt --classes=model_data/yolo3/coco/coco_classes.txt --image
Input image filename:images/pics/dog.jpg
```

## Run glob pattern

```
python3 test_image.py --model=model_data/yolo3/coco/yolo.h5 --anchors=model_data/yolo3/coco/yolo_anchors.txt --classes=model_data/yolo3/coco/coco_classes.txt -i=images/pics/*jpg

python3 test_image.py --model=model_data/mrcnn/coco/mask_rcnn_coco.h5 --classes=model_data/mrcnn/coco/classes.txt -i=images/pics/eagle.jpg -n=mrcnn
```

## Prepare Dataset

```
mkdir mscoco2017
cd mscoco2017/
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
python3 coco_annotation.py
less train.txt
mscoco2017/train2017/000000558840.jpg 199,200,276,270,52 325,104,358,209,39 168,90,199,178,39 1,87,35,262,41 346,1,638,344,0 239,42,258,118,39 409,215,480,265,44 0,1,93,161,0 276,13,307,74,39 3,263,362,419,60 413,201,485,257,44
mscoco2017/train2017/000000200365.jpg 234,317,383,355,52 239,347,399,404,52 296,388,297,388,52 251,333,376,355,52 128,192,639,473,60 0,36,562,479,1 131,0,639,248,2 1,1,131,58,2 463,202,562,372,41
mscoco2017/train2017/000000495357.jpg 337,244,403,310,16 255,257,436,370,3 509,215,556,239,26 22,206,43,229,26 354,162,384,211,26 481,107,600,351,0 445,196,502,271,0 390,193,464,280,0 381,127,414,254,0 356,143,392,268,0 233,148,271,221,0 197,142,238,284,0 15,133,65,311,0 100,149,148,303,0 109,167,156,239,0 591,121,620,204,0 186,147,214,248,0 241,197,274,248,0 277,122,575,332,0
filepath x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id.... 
```

## Train Model

```
python3 scripts/data_split.py -a=train.txt -n=10 -c=model_data/yolo3/coco/coco_classes.txt
python3 keras-yolo3/train.py -m model_data/yolo3/coco/yolo.h5 -c=model_data/yolo3/coco/coco_classes.txt -t=_000/train.txt
```

## Evaluate Model

```
python3 test_image.py -t=_000/test.txt
python3 scripts/compute_mAP_IoU.py results/yolo.h5_001/ _001/ground-truth/
```

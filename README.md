# object-detection

## How to Use

## Set CUDA_VISIBLE_DEVICES
CPU
```
export CUDA_DEVICE_ORDER=PCI_BUS_ID; export CUDA_VISIBLE_DEVICES=-1;
```
GPU
```
export CUDA_DEVICE_ORDER=PCI_BUS_ID; export CUDA_VISIBLE_DEVICES=0;
```
## Prepare package
[InstallCUDA](/InstallCUDA.md)
```
git clone https://github.com/soatcoap/keras-yolo3
ln -s keras-yolo3/yolo3 .
pip3 install -r requirements.txt
```
```
git clone https://github.com/matterport/Mask_RCNN.git
ln -s Mask_RCNN/mrcnn .
pip3 install -r Mask_RCNN/requirements.txt
```
```
git clone https://github.com/see--/keras-centernet.git
ln -s keras-centernet/keras_centernet .
```

## Prepare Model
```
wget https://pjreddie.com/media/files/yolov3.weights
python3 keras-yolo3/convert.py cfg/yolov3.cfg yolov3.weights model_data/yolo3/coco/yolo.h5
```

```
wget https://pjreddie.com/media/files/yolov3-openimages.weights
python3 keras-yolo3/convert.py cfg/yolov3-openimages.cfg yolov3-openimages.weights model_data/yolo3/openimage/yolov3-openimages.h5
```

```
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 -P model_data/mrcnn/coco/
```
```
wget https://github.com/see--/keras-centernet/releases/download/0.1.0/ctdet_coco_hg.hdf5 -P model_data/keras-centernet/coco/
```

## Run

```
python3 test_video.py --model=model_data/yolo3/coco/yolo.h5 --anchors=model_data/yolo3/coco/anchors.txt --classes=model_data/yolo3/coco/classes.txt
```
```
python3 test_video.py --model=model_data/yolo3/coco/yolo.h5 --anchors=model_data/yolo3/coco/anchors.txt --classes=model_data/yolo3/coco/classes.txt --image
Input image filename:images/pics/dog.jpg
```

## Run glob pattern

```
python3 test_image.py --model=model_data/yolo3/coco/yolo.h5 --anchors=model_data/yolo3/coco/anchors.txt --classes=model_data/yolo3/coco/classes.txt -i=images/pics/*jpg
```
```
python3 test_image.py --model=model_data/yolo3/openimage/yolov3-openimages.h5 --anchors=model_data/yolo3/coco/anchors.txt --classes=model_data/yolo3/openimage/classes.txt -i=images/pics/*jpg
```
```
python3 test_image.py --model=model_data/mrcnn/coco/mask_rcnn_coco.h5 --classes=model_data/mrcnn/coco/classes.txt -i=images/pics/eagle.jpg -n=mrcnn
```
```
python3 test_image.py --model=model_data/keras-centernet/coco/ctdet_coco_hg.hdf5 --classes=model_data/keras-centernet/coco/classes.txt -i=images/pics/*.jpg -n=keras-centernet
```

## Prepare Dataset(mscoco)
```
mkdir mscoco2017
cd mscoco2017/
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
cd ..
```

```
python3 keras-yolo3/coco_annotation.py
mv train.txt data_labels/bbox/coco/
```
```
python3 scripts/augment.py -a=data_labels/bbox/coco/train.txt -ho -n=20
```
```
less data_labels/bbox/coco/train.txt
mscoco2017/train2017/000000558840.jpg 199,200,276,270,52 325,104,358,209,39 168,90,199,178,39 1,87,35,262,41 346,1,638,344,0 239,42,258,118,39 409,215,480,265,44 0,1,93,161,0 276,13,307,74,39 3,263,362,419,60 413,201,485,257,44
mscoco2017/train2017/000000200365.jpg 234,317,383,355,52 239,347,399,404,52 296,388,297,388,52 251,333,376,355,52 128,192,639,473,60 0,36,562,479,1 131,0,639,248,2 1,1,131,58,2 463,202,562,372,41
mscoco2017/train2017/000000495357.jpg 337,244,403,310,16 255,257,436,370,3 509,215,556,239,26 22,206,43,229,26 354,162,384,211,26 481,107,600,351,0 445,196,502,271,0 390,193,464,280,0 381,127,414,254,0 356,143,392,268,0 233,148,271,221,0 197,142,238,284,0 15,133,65,311,0 100,149,148,303,0 109,167,156,239,0 591,121,620,204,0 186,147,214,248,0 241,197,274,248,0 277,122,575,332,0
filepath x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id.... 
```
```
python3 scripts/coco_json_to_mrcnn_txt.py
less data_labels/polygon/coco/train_list.txt
mscoco2017/train2017/000000558840.jpg [[[239,260,222,270,199,253,213,227,259,200,274,202,277,210,249,253,237,264,242,261,228,271]],53] [[[357,210,338,209,327,204,325,164,329,127,326,108,333,104,348,104,358,108,358,130]],40] [[[x1,y1,x2,y2,x3,y3,...][x1,y1,x2,y2,x3,y3,...][...]...],class_id]]....
cd data_labels/polygon/coco/
head train_list.txt -n 100 >  train_list_100.txt
```

## Prepare Dataset(ead2019 object detection)
1. Download ead2019 dataset(trainingData_detection.zip) to downloads/ead2019 directory and unzip it.
2. Convert to a format that can be used for learning.
```
python3 scripts/ead2019_to_train.py -e=downloads/ead2019/trainingData_detection -o=data_labels/bbox/ead2019
```

## Train Model

```
python3 scripts/data_split.py -a=data_labels/bbox/coco/train.txt  -af=polygon -n=10 -c=model_data/yolo3/coco/classes.txt
```
```
python3 train_yolo.py -m=model_data/yolo3/coco/yolo.h5 -t=data_labels/bbox/coco_000/train.txt -c=model_data/yolo3/coco_000/classes.txt
```
```
python3 train_mrcnn.py train -a=data_labels/polygon/coco/train_list_100.txt  -w=model_data/mrcnn/coco/mask_rcnn_coco.h5
```

## Evaluate Model

```
python3 test_image.py -t=data_labels/bbox/coco_000/test.txt  --anchors=model_data/yolo3/coco/anchors.txt --classes=model_data/yolo3/coco/classes.txt -m=model_data/yolo3/coco/yolo.h5
```
```
python3 scripts/calculate_mAP_IoU.py -r results/yolo/coco_000/yolo.h5_000/ -g data_labels/bbox/coco_000/ground-truth/
cat results/yolo/coco_000/yolo.h5_000/mAP_IOU/results.txt | grep _25
```
```
python3 test_image.py -t=data_labels/bbox/coco_000/test.txt --model=model_data/keras-centernet/coco/ctdet_coco_hg.hdf5 --classes=model_data/keras-centernet/coco/classes.txt -n=keras-centernet
```
```
python3 scripts/calculate_mAP_IoU.py -r results/keras-centernet/coco_000/ctdet_coco_hg.hdf5_001/ -g  data_labels/bbox/coco_000/ground-truth/
cat results/keras-centernet/coco_000/ctdet_coco_hg.hdf5_001//mAP_IOU/results.txt | grep _25
```

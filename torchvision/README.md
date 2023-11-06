# torchvision
# docker build and run
```
./build.sh
./run_nvidia_docker.sh
```
# train
モデルを学習時と予測時に-mを用いてfaster-rcnn, ssd, retinanet, fcosに適宜変更してください。
出力の1行目がtrain loss, 2行目がvalidation lossとなっています。logは必要であれば手動でどこかに保存してください
```
python3 torchvision/train.py -a=data_labels/bbox/ead2019_000/train.txt -c=data_labels/bbox/ead2019_000/classes.txt -e=70 -m=faster-rcnn
```
# predict
学習で得られた重みを-wで指定して予測を行います。大体それぞれ50epoch目ぐらいの重みでmAPが最大となると思いますがいくつか試す必要があります。ただし、学習時のvalidation lossが低いときmAPが高くなるとは限らないようです
```
python3 torchvision/predict.py -a=data_labels/bbox/ead2019_000/test.txt -c=data_labels/bbox/ead2019_000/classes.txt -w=model_data/faster-rcnn/02280953/weight-45.pth
```
# calculate mAP
mAP等を計算 予測で作成されたディレクトリを-rで指定してください。
```
python3 scripts/calculate_mAP_IoU.py -r=results/faster-rcnn/02281606 -g=data_labels/bbox/ead2019_000/ground-truth -iou=0.5 -ios=1.1
```
# combine predictions
results/combinedに二つのpredictionを合わせたpredicitonを作成します。何を組み合わせるかはcombine.py内でyolo_classes とfaster_rcnn_classesのリストを書き換えて指定します。
```
python3 torchvision/combine.py -yr=results/yolo/ead2019_000/000_000 -fr=results/faster-rcnn/02281606
```

# list="retinanet ssd"
# list="faster-rcnn"
list="faster-rcnn ssd fcos retinanet"
# list="retinanet"
for n in $list; do
    echo $n
  for i in {0..4}
#   for i in {3..4}
#   for i in {0..0}
  do
    echo "${i}"
    model="${n}_kfold_000_00${i}_minvallosswhole"
    # model="${n}_kfold_000_00${i}_final"
    # if [ $n -eq "yolo3_ead2019" ] ; then
    #     model="${n}_kfold_000_00${i}_final"
    # fi
    # echo "${model}"
    # sc="python3 scripts/calculate_mAP_IoU.py -r results/${model}/000 -g data_labels/bbox/ead2019/ead2019_kfold_000/00${i}/ground-truth -th 0.${t}"
    sc="python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/00${i}/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/${model}.pth -n 1.1 -s 0.009 "
    echo $sc
    # $sc
    
    # echo python3 scripts/calculate_mAP_IoU.py -r results/model/000 -g data_labels/bbox/ead2019/ead2019_kfold_000/00${i}/ground-truth
    # python3 scripts/calculate_mAP_IoU.py -r results/model/000 -g data_labels/bbox/ead2019/ead2019_kfold_000/00${i}/ground-truth
  done
done

# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/000/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/faster-rcnn_kfold_000_000_minvallosswhole.pth -n 0.3 -s 0.1 
# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/001/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/faster-rcnn_kfold_000_001_minvallosswhole.pth -n 0.3 -s 0.1 
# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/002/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/faster-rcnn_kfold_000_002_minvallosswhole.pth -n 0.3 -s 0.1
# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/003/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/faster-rcnn_kfold_000_003_minvallosswhole.pth -n 0.3 -s 0.1
# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/004/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/faster-rcnn_kfold_000_004_minvallosswhole.pth -n 0.3 -s 0.1

# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/000/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/fcos_kfold_000_000_minvallosswhole.pth -n 0.3 -s 0.1 
# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/001/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/fcos_kfold_000_001_minvallosswhole.pth -n 0.3 -s 0.1 
# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/002/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/fcos_kfold_000_002_minvallosswhole.pth -n 0.3 -s 0.1
# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/003/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/fcos_kfold_000_003_minvallosswhole.pth -n 0.3 -s 0.1
# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/004/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/fcos_kfold_000_004_minvallosswhole.pth -n 0.3 -s 0.1

# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/000/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/ssd_kfold_000_000_minvallosswhole.pth -n 0.3 -s 0.1 
# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/001/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/ssd_kfold_000_001_minvallosswhole.pth -n 0.3 -s 0.1 
# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/002/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/ssd_kfold_000_002_minvallosswhole.pth -n 0.3 -s 0.1
# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/003/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/ssd_kfold_000_003_minvallosswhole.pth -n 0.3 -s 0.1
# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/004/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/ssd_kfold_000_004_minvallosswhole.pth -n 0.3 -s 0.1

# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/000/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/retinanet_kfold_000_000_minvallosswhole.pth -n 0.3 -s 0.1
# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/001/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/retinanet_kfold_000_001_minvallosswhole.pth -n 0.3 -s 0.1 
# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/002/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/retinanet_kfold_000_002_minvallosswhole.pth -n 0.3 -s 0.1
# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/003/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/retinanet_kfold_000_003_minvallosswhole.pth -n 0.3 -s 0.1
# python3 torchvision/predict.py -a=data_labels/bbox/ead2019/ead2019_kfold_000/004/test.txt -c=data_labels/bbox/ead2019/classes.txt -w model_data/all/retinanet_kfold_000_004_minvallosswhole.pth -n 0.3 -s 0.1

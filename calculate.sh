
list="yolo3_ead2019 faster-rcnn "
list="ssd"
list="retinanet"
list="fcos"
list="yolo3_ead2019 faster-rcnn ssd fcos retinanet"
# list="faster-rcnn"
# list="yolo3_ead2019"
for n in $list; do
    echo $n
  for i in {0..4}
#   for i in {3..4}
  # for i in {0..0}
  do
    echo "${i}"
    model="${n}_kfold_000_00${i}_minvallosswhole"
    # model="${n}_kfold_000_00${i}_final"
    if [ $n == "yolo3_ead2019" ] ; then
        model="${n}_kfold_000_00${i}_final"
    fi
    # echo "${model}"
    # sc="python3 scripts/calculate_mAP_IoU.py -r results/${model}/00${i} -g data_labels/bbox/ead2019/ead2019_kfold_000/00${i}/ground-truth -th 0.01 -iou 0.3"
    # echo $sc
    # $sc
    for t in {1..9}; do
    # for t in {5..5}; do
    # for t in {9..9}; do
    # for t in {1..1}; do
        # echo "${t}"
        sc="python3 scripts/calculate_mAP_IoU.py -r results/${model}/00${i} -g data_labels/bbox/ead2019/ead2019_kfold_000/00${i}/ground-truth -th 0.${t} -iou 0.3"
        echo $sc
        $sc

        # echo python3 scripts/calculate_mAP_IoU.py -r results/model/000 -g data_labels/bbox/ead2019/ead2019_kfold_000/00${i}/ground-truth
        # python3 scripts/calculate_mAP_IoU.py -r results/model/000 -g data_labels/bbox/ead2019/ead2019_kfold_000/00${i}/ground-truth
    done
  done
done

# python3 scripts/calculate_mAP_IoU.py -r results/ead2019_kfold_000_000/000 -g data_labels/bbox/ead2019/ead2019_kfold_000/000/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/ead2019_kfold_000_001/001 -g data_labels/bbox/ead2019/ead2019_kfold_000/001/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/ead2019_kfold_000_002/002 -g data_labels/bbox/ead2019/ead2019_kfold_000/002/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/ead2019_kfold_000_003/003 -g data_labels/bbox/ead2019/ead2019_kfold_000/003/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/ead2019_kfold_000_004/004 -g data_labels/bbox/ead2019/ead2019_kfold_000/004/ground-truth

# python3 scripts/calculate_mAP_IoU.py -r results/faster-rcnn_kfold_000_000_minvallosswhole/000 -g data_labels/bbox/ead2019/ead2019_kfold_000/000/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/faster-rcnn_kfold_000_001_minvallosswhole/001 -g data_labels/bbox/ead2019/ead2019_kfold_000/001/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/faster-rcnn_kfold_000_002_minvallosswhole/002 -g data_labels/bbox/ead2019/ead2019_kfold_000/002/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/faster-rcnn_kfold_000_003_minvallosswhole/003 -g data_labels/bbox/ead2019/ead2019_kfold_000/003/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/faster-rcnn_kfold_000_004_minvallosswhole/004 -g data_labels/bbox/ead2019/ead2019_kfold_000/004/ground-truth

# python3 scripts/calculate_mAP_IoU.py -r results/ssd_kfold_000_000_minvallosswhole/000 -g data_labels/bbox/ead2019/ead2019_kfold_000/000/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/ssd_kfold_000_001_minvallosswhole/001 -g data_labels/bbox/ead2019/ead2019_kfold_000/001/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/ssd_kfold_000_002_minvallosswhole/002 -g data_labels/bbox/ead2019/ead2019_kfold_000/002/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/ssd_kfold_000_003_minvallosswhole/003 -g data_labels/bbox/ead2019/ead2019_kfold_000/003/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/ssd_kfold_000_004_minvallosswhole/004 -g data_labels/bbox/ead2019/ead2019_kfold_000/004/ground-truth

# python3 scripts/calculate_mAP_IoU.py -r results/fcos_kfold_000_000_minvallosswhole/000 -g data_labels/bbox/ead2019/ead2019_kfold_000/000/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/fcos_kfold_000_001_minvallosswhole/001 -g data_labels/bbox/ead2019/ead2019_kfold_000/001/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/fcos_kfold_000_002_minvallosswhole/002 -g data_labels/bbox/ead2019/ead2019_kfold_000/002/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/fcos_kfold_000_003_minvallosswhole/003 -g data_labels/bbox/ead2019/ead2019_kfold_000/003/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/fcos_kfold_000_004_minvallosswhole/004 -g data_labels/bbox/ead2019/ead2019_kfold_000/004/ground-truth


# python3 scripts/calculate_mAP_IoU.py -r results/retinanet_kfold_000_000_minvallosswhole/000 -g data_labels/bbox/ead2019/ead2019_kfold_000/000/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/retinanet_kfold_000_001_minvallosswhole/001 -g data_labels/bbox/ead2019/ead2019_kfold_000/001/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/retinanet_kfold_000_002_minvallosswhole/002 -g data_labels/bbox/ead2019/ead2019_kfold_000/002/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/retinanet_kfold_000_003_minvallosswhole/003 -g data_labels/bbox/ead2019/ead2019_kfold_000/003/ground-truth
# python3 scripts/calculate_mAP_IoU.py -r results/retinanet_kfold_000_004_minvallosswhole/004 -g data_labels/bbox/ead2019/ead2019_kfold_000/004/ground-truth

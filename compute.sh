
network = ["ead2019", "faster-rcnn", "ssd", "fcos", "retinanet"]
# list="retinanet ssd"
# list="faster-rcnn"
list="faster-rcnn ssd fcos retinanet"
# list="yolo3_ead2019"
for n in $list; do
    echo $n
  # for i in {0..0}
  for i in {0..4}
#   for i in {3..4}
  do
    echo "${i}"
    model="${n}_kfold_000_00${i}_minvallosswhole"
    # model="${n}_kfold_000_00${i}_final"

    # if [ $n -eq "yolo3_ead2019" ] ; then
    #     model="${n}_kfold_000_00${i}_final"
    # fi
    # echo "${model}"

    for t in 0.09 0.8; do
        # echo "${t}"
        sc="python scripts/compute_mAP_IoU.py results/${model}/00${i}/predictions data_labels/bbox/ead2019/ead2019_kfold_000/00${i}/ground-truth/ results/${model}/00${i}/compute_th${t} res.json ${t}"
        echo $sc
        $sc
    done

  done
done

# python scripts/compute_mAP_IoU.py results/test1/004/predictions results/test1/004/ground-truth results/test1/004/compute res.json 0.1

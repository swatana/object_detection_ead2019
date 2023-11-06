import argparse
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import os
import datetime
import shutil

from tvutils import load_anno,TestDataset

def detect_img():
    anno_file_path = FLAGS.annotation
    anno_file_dir = os.path.basename(os.path.dirname(anno_file_path))

    anno_file_name = os.path.splitext(os.path.basename(anno_file_path))[0]
    class_file_path = FLAGS.classes_path
    model_name = FLAGS.model
    if model_name == None:
        weight_file_path = FLAGS.weight
        weight_file_name = os.path.splitext(os.path.basename(weight_file_path))[0]
    else:
        weight_file_name = model_name

    device = FLAGS.device
    device = FLAGS.device
    scorethresh = FLAGS.scorethresh
    nms_thresh = FLAGS.nms_thresh
    classes = ["BG"]
    with open(class_file_path) as f:
        lines = f.readlines()
        for line in lines:
            classes.append(line.replace("\n",""))
            
    date = datetime.datetime.now().strftime('%m%d%H%M')
    results_dir = f'./results/{weight_file_name}/{anno_file_dir}/'
    if not os.path.exists(results_dir): 
        os.makedirs(f'{results_dir}/images')
        os.makedirs(f'{results_dir}/predictions')
    shutil.copyfile(class_file_path,f'{results_dir}/classes.txt')
    shutil.copyfile(anno_file_path,f'{results_dir}/test.txt')
    
    if model_name == "faster-rcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(rpn_nms_thresh=nms_thresh,
                    box_nms_thresh=nms_thresh,
                    box_detections_per_img=10000,
                    rpn_pre_nms_top_n_test=10000,
                    rpn_post_nms_top_n_test=10000)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))
    elif model_name == "ssd":
        model = torchvision.models.detection.ssd.ssd300_vgg16(num_classes=len(classes), detections_per_img=1000, nms_thresh=nms_thresh, iou_thresh=nms_thresh)
    elif model_name == "retinanet":
        model = torchvision.models.detection.retinanet_resnet50_fpn(num_classes=len(classes), nms_thresh=nms_thresh, detections_per_img=10000)
    elif model_name == "fcos":
        model = torchvision.models.detection.fcos_resnet50_fpn(num_classes=len(classes), nms_thresh=nms_thresh, detections_per_img=10000)
    # model.load_state_dict(torch.load(weight_file_path))
    with open(anno_file_path) as f:
        image_dir = os.path.dirname(f.readline().split(" ")[0])

    df = load_anno(anno_file_path)
    test = TestDataset(df,image_dir)
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    dataloader = torch.utils.data.DataLoader(test,batch_size=1, shuffle=False, collate_fn=collate_fn)

    if model_name == "faster-rcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(rpn_nms_thresh=1.0,
                    box_nms_thresh=1.0,
                    box_detections_per_img=10000,
                    rpn_pre_nms_top_n_test=10000,
                    rpn_post_nms_top_n_test=10000)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))
    elif model_name == "ssd":
        model = torchvision.models.detection.ssd.ssd300_vgg16(num_classes=len(classes), nms_thresh=1.0, detections_per_img=10000)
    elif model_name == "retinanet":
        model = torchvision.models.detection.retinanet_resnet50_fpn(num_classes=len(classes), nms_thresh=1.0, detections_per_img=10000)
    elif model_name == "fcos":
        model = torchvision.models.detection.fcos_resnet50_fpn(num_classes=len(classes), nms_thresh=1.0, detections_per_img=10000)
    elif model_name == "yolov4":
        from torchvision.models.detection import yolov4, YOLOV4_Weights
        model = yolov4(weights=YOLOV4_Weights.DEFAULT)
    else:
        if device == 'cuda':
            model.load_state_dict(torch.load(weight_file_path))
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
        elif device == 'mps':
            model = torch.load(weight_file_path, map_location=torch.device('mps'))
            device = torch.device('mps')
            # model.mps()
        else:
            model = torch.load(weight_file_path, map_location=torch.device('cpu'))
            # model.load_state_dict(torch.load(weight_file_path, map_location=torch.device('cpu')))
            device = torch.device('cpu') 
    model.to(device)
    model.eval()

    for images,image_ids in dataloader:
        output = ""
        images = list(image.to(device) for image in images)
        with torch.no_grad():
            prediction = model(images)
        # print(prediction)
        # print(len(prediction))
        # print(prediction[0])
        # print(prediction[0]['boxes'])
        for j in range(len(images)):
            # print("j", j)
            imgfile=image_dir+'/'+image_ids[j]+'.jpg'
            img = cv2.imread(imgfile)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            num_boxs=0
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i,box in enumerate(prediction[j]['boxes']):
                score = prediction[j]['scores'][i].cpu().numpy()
                if score > scorethresh:
                    score = round(float(score),2)
                    cat = prediction[j]['labels'][i].cpu().numpy()
                    txt = '{} {}'.format(classes[int(cat)], str(score))
                    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
                    box=box.cpu().numpy().astype('int')
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255,255,255) , 1)
                    cv2.rectangle(img,(box[0], box[1] - cat_size[1] - 2),(box[0] + cat_size[0], box[1] - 2), (255,255,255), -1)
                    cv2.putText(img, txt, (box[0], box[1] - 2), font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                    num_boxs += 1
                    output += "{class_name}\t{score}\t{coordinates}\n".format(
                            class_name=classes[int(cat)],
                            score=score,
                            coordinates = "{0}\t{1}\t{2}\t{3}".format(
                                box[0], box[1], box[2], box[3]),
                        )
            
            cv2.imwrite(f"{results_dir}/images/{image_ids[j]}.jpg",cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            with open(f"{results_dir}/predictions/{image_ids[j]}.txt","w") as f:
                print(output, end="", file=f)
                    

FLAGS=None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        "-d", "--device", nargs='?', type=str, default="cpu",
        help="cpu"
    )
    parser.add_argument(
        "-a", "--annotation", nargs='?', type=str, default=None,
        help="Annotation path"
    )
    parser.add_argument(
        "-w", "--weight", nargs='?', type=str, default=None,
        help="Weight path"
    )
    parser.add_argument(
        "-c", "--classes_path", nargs='?', type=str, default=None,
        help="Weight path"
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        choices=[
            'ssd',
            'faster-rcnn',
            'retinanet',
            'fcos',
            'yolov4'],
        help='model name(faster-rcnn, ssd ,fcos or retinanet)',
        default=None,
    )
    parser.add_argument(
        "-n", "--nms_thresh", nargs='?', type=str, default=1.1,
        help="Nms thresh"
    )
    parser.add_argument(
        "-s", "--scorethresh", nargs='?', type=float, default=0.01,
        help="Score thresh"
    )
    FLAGS = parser.parse_args()
    detect_img()

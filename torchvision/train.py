import torch
import torchvision
import argparse
import os
import datetime
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from statistics import mean

import shutil

from tvutils import load_anno,Dataset


def get_unused_dir_num(path='./model_data'):
    dir_list = os.listdir(path=path)
    for i in range(1000):
        search_dir_name = '%03d' % i
        if search_dir_name not in dir_list:
            return search_dir_name
    raise NotFoundError('Error')

def train():
    anno_file_path = FLAGS.annotation
    classes_file_path = FLAGS.classes
    modelname = FLAGS.model
    num_epochs = FLAGS.epochs
    out = FLAGS.out
    device_name = FLAGS.device_name
    with open(anno_file_path) as f:
        image_dir = os.path.dirname(f.readline().split(" ")[0])
    # now = datetime.datetime.now().strftime('%m%d%H%M')
    # modelpath = f'./model_data/{modelname}/{now}'
    if out == False:
        modelpath = os.path.join(f'./model_data/{modelname}/', get_unused_dir_num())
    else:
        modelpath = os.path.join(f'./model_data/{modelname}/', out)
    os.makedirs(modelpath,exist_ok=True)
    

    shutil.copyfile(anno_file_path, os.path.join(modelpath, "train.txt"))
    shutil.copyfile(classes_file_path, os.path.join(modelpath, "classes.txt"))

    classes = ["BG"]
    with open(classes_file_path) as f:
        lines = f.readlines()
        for line in lines:
            classes.append(line.replace("/n",""))
    df = load_anno(anno_file_path)
    train_df = df.sample(frac=0.8,random_state=60)
    val_df = df.drop(train_df.index)
    torch.manual_seed(1000)
    train = Dataset(train_df,image_dir)
    val = Dataset(val_df,image_dir)
    def collate_fn(batch):
        return tuple(zip(*batch))
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=1, shuffle=True, collate_fn=collate_fn)
    if modelname == "faster-rcnn" or modelname == "fasterrcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
    if modelname == "fasterrcnnv2":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
    if modelname == "ssd":
        model = torchvision.models.detection.ssd.ssd300_vgg16(num_classes=len(classes), weights=None,)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
    if modelname == "retinanet":
        model = torchvision.models.detection.retinanet_resnet50_fpn(num_classes=len(classes))
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
    if modelname == "fcos":
        model = torchvision.models.detection.fcos_resnet50_fpn(num_classes=len(classes))
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
    if modelname == "yolov8":

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)
        # from ultralytics import YOLO
        # model = YOLO("yolov8n.yaml")
        # import nn
        # model = nn.yolo_v8_n(len(classes))
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
    if modelname == "yolov5":
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
    if modelname == "yolov4":
        model = torchvision.models.detection.yolov4(num_classes=len(classes))
        # model = torchvision.models.detection.YOLOV7Networka(num_classes=(len(classes)))
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
   
    params = [p for p in model.parameters() if p.requires_grad]
    
    
    if device_name == 'cuda':
        torch.cuda.empty_cache()
        device = torch.device('cuda')
        model.cuda()
    elif device_name == 'mps':
        device = torch.device('mps')
        model.to('mps')
        # model.cuda()
        # model.mps()
    else:
        device = torch.device('cpu') 
    
    model.train()
    min = float("inf")
    for epoch in range(num_epochs):
        loss_list = []
        val_loss_list = []
        loss_dicts = {}
        for i, batch in enumerate(train_dataloader):
            images, targets, image_ids = batch
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            for a, b in zip(loss_dict.keys(),loss_dict.values()):
                if i == 0:
                    loss_dicts[a] = [b.item()]
                else:
                    loss_dicts[a].append(b.item()) 

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            loss_list.append(loss_value)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        print(f"{epoch+1}/{num_epochs} loss:{mean(loss_list)}",end="")
        for a in loss_dicts.keys():
            print(f", {a}:{mean(loss_dicts[a])}",end="")
        print("")

        loss_dicts = {}
        for i, batch in enumerate(val_dataloader):
            images, targets, image_ids = batch
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict= model(images, targets)
            for a, b in zip(loss_dict.keys(),loss_dict.values()):
                if i == 0:
                    loss_dicts[a] = [b.item()]
                else:
                    loss_dicts[a].append(b.item()) 
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            val_loss_list.append(loss_value)

        print(f"{epoch+1}/{num_epochs} loss:{mean(val_loss_list)}",end="")
        for a in loss_dicts.keys():
            print(f", {a}:{mean(loss_dicts[a])}",end="")
        print("")

        if min>mean(val_loss_list):
            torch.save(model.state_dict(), f'{modelpath}/weight-minvalloss.pth')
            torch.save(model, f'{modelpath}/weight-minvallosswhole.pth')
            # torch.save(model.state_dict(), f'model_data/all/{modelname}_{out}_minvalloss.pth') 
            # torch.save(model, f'model_data/all/{modelname}_{out}_minvallosswhole.pth') 
            min=mean(val_loss_list)
        if (epoch+1)%10 == 0:
            torch.save(model, f'{modelpath}/weight-{epoch+1}whole.pth') 
            torch.save(model.state_dict(), f'{modelpath}/weight-{epoch+1}.pth')  
    # torch.save(model, f'model_data/all/{modelname}_{out}_{epoch+1}whole.pth') 
    # torch.save(model.state_dict(), f'model_data/all/{modelname}_{out}_{epoch+1}.pth') 


FLAGS=None
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        "-d", "--device_name", nargs='?', type=str, default="cuda",
        help="cuda"
    )
    parser.add_argument(
        "-a", "--annotation", nargs='?', type=str, default=None,
        help="Annotation path",
        required=True,
    )
    parser.add_argument(
        "-c", "--classes", nargs='?', type=str, default=None,
        help="classes path",
        required=True,
    )
    parser.add_argument(
        "-o", "--out", nargs='?', type=str, default=None,
        help="output dir",
        required=True,
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        choices=[
            'ssd',
            'faster-rcnn',
            'fasterrcnn',
            'fasterrcnnv2',
            'retinanet',
            'fcos',
            'yolov8',
            'yolov5',
            'yolov4'],
        help='model name(faster-rcnn, ssd, fcos or retinanet)',
        default="fasterrcnnv2",
    )
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help='number of epochs',
        default="50",
    )
    FLAGS = parser.parse_args()
    train() 

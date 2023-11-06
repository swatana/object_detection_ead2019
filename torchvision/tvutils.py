import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from glob import glob

def load_anno(path):
    with open(path) as f:
        lines=f.readlines()
    df = pd.DataFrame(columns=["image_id","xmin", "ymin", "xmax", "ymax", "class"])
    for line in lines:
        image_id = line.split("/")[-1].split(".")[0]
        bboxs = line.replace("\n","").split(" ")[1:]
        for bbox in bboxs:
            tmp = pd.Series(map(float,bbox.split(",")), index=["xmin", "ymin", "xmax", "ymax", "class"])
            tmp["image_id"] = image_id
            df=pd.concat([df,pd.DataFrame([tmp])],ignore_index=True)
    df["class"] = df["class"] + 1
    return df 


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dir):
        super().__init__()
        self.image_ids = df["image_id"].unique()
        self.df = df
        self.image_dir = image_dir
        
    def __getitem__(self, index):
        transform = transforms.Compose([
                                        transforms.ToTensor()
        ])
        image_id = self.image_ids[index]
        image = Image.open(f"{self.image_dir}/{image_id}.jpg")
        image = transform(image)
        records = self.df[self.df["image_id"] == image_id]
        boxes = torch.tensor(records[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = torch.tensor(records["class"].values, dtype=torch.int64)
        iscrowd = torch.zeros((records.shape[0], ), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"]= labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd
        return image, target, image_id
        
    def __len__(self):
        return self.image_ids.shape[0]

class TestDataset(torch.utils.data.Dataset):
    def __init__(self,df,image_dir):
        
        super().__init__()
        self.image_ids = df["image_id"].unique()
        self.df = df
        self.image_dir = image_dir
    
        
    def __getitem__(self, index):
        transform = transforms.Compose([
                                        transforms.ToTensor()
        ])
        image_id = self.image_ids[index]
        image = Image.open(f"{self.image_dir}/{image_id}.jpg")
        image = transform(image)
        records = self.df[self.df["image_id"] == image_id]
        boxes = torch.tensor(records[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = torch.tensor(records["class"].values, dtype=torch.int64)
        iscrowd = torch.zeros((records.shape[0], ), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"]= labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd
        return image, image_id
    
    def __len__(self):
        return len(self.image_ids)

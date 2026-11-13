import torch
from torchvision.datasets import VOCDetection
from pathlib import Path
import os
import cv2
import numpy as np
from pprint import  pprint

from torchvision.transforms import ToTensor


class VOCDataset(VOCDetection):
    def __init__(self,root,year, image_set,download = False, transform = None):
        super().__init__(root,year, image_set, download,transform)
        self.transform = transform

    def __getitem__(self,index):
        classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike","person","pottedplant","sheep", "sofa", "train", "tvmonitor"]
        image, data = super().__getitem__(index)
        if self.transform:
            image = self.transform(image)
        bndbox = data["annotation"]["object"]
        # print(list(bndbox[0]["bndbox"].values()))
        target = {}
        boxes = []
        id_class = []
        for i in range (len(bndbox)):
            d = {}
            bb = bndbox[i]["bndbox"]
            boxes.append([float(bb['xmin']),float(bb['ymin']),float(bb['xmax']),float(bb['ymax'])])

            name = bndbox[i]["name"]

            if name in classes:
                id_class.append(classes.index(name)+1)
            else:
                id_class.append(0)

        boxes = torch.tensor(boxes,dtype=torch.float32)
        target['boxes']= boxes
        id_class = torch.tensor(id_class,dtype=torch.int64)
        target['labels'] = id_class

        # print(type(id_class))
        # print(type(boxes))
        # print(id_class.size())
        # print(boxes.size())
        return image, target

if __name__ == '__main__':
    transform = ToTensor()
    data = VOCDataset(root="dataset/VOCdevkit", year= "2012",image_set="train", transform= transform)
    image, target = data[1001]
    print(type(target["boxes"]))
    device = torch.device("cuda")
    print(type(image))
    # pprint(image)
    # pprint(target["annotation"])

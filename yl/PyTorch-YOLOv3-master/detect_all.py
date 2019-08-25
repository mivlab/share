from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from torchvision import transforms, models
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    #parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    #parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    #parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--image_folder", type=str, default="data/minesamples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_2.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")

    parser.add_argument("--conf_thres", type=float, default=0.98, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    print("\nPerforming object detection:")
    prev_time = time.time()
    files = os.listdir(opt.image_folder)
    for file in files:
        print(file)
        img_topleft = []  # Stores image paths
        img_detections = []  # Stores detections for each image index
        img_path = os.path.join(opt.image_folder, file)
        #img_ = Image.open(img_path).convert('RGB')
        img_ = cv2.imread(img_path)
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        h, w, c = img_.shape
        if h < opt.img_size or w < opt.img_size:
            print('image resolution is too low')
            continue
        step = (opt.img_size * 4) // 5
        for row in range(0, h - opt.img_size + 1, step):
            for col in range(0, w - opt.img_size + 1, step):
                patch = img_[row:row + opt.img_size, col:col + opt.img_size, :].copy()
                img_tensor = transforms.ToTensor()(patch)
                img_tensor = img_tensor.unsqueeze(0)
                if torch.cuda.is_available():
                    detections = model(Variable(img_tensor.cuda()))
                else:
                    detections = model(Variable(img_tensor))
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
                img_topleft.extend([[col, row]])
                img_detections.extend(detections)
                print(detections)

        img = cv2.imread(img_path)
        for img_i, (start, detections) in enumerate(zip(img_topleft, img_detections)):
            if detections is not None:
                detections1 = rescale_boxes(detections, opt.img_size, (opt.img_size, opt.img_size))
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections1:
                    cv2.rectangle(img, (x1 + start[0], y1 + start[1]), (x2 + start[0], y2 + start[1]), (255, 0, 0), 2)
        if img.shape[0] > 1920 or img.shape[1] > 1080:
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        cv2.imshow('result', img)
        cv2.waitKey()

    exit()

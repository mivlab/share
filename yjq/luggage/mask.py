# coding: utf-8
import numpy as np
import cv2
import json
import os
import random
with open(r'D:\work\data\MVB_train\Info\train.json', 'r', encoding='utf-8') as f:
  for line in f:
    json_dict = json.loads(line)
  for item in json_dict['image']:
    image_name = item['image_name']
    mask = item['mask']
    print('%s:%s' %(image_name,mask))
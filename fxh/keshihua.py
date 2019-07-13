import torch
import cv2 as cv
import numpy as np
#from PIL import Image
import csv
#import matplotlib.pyplot as plt
import random
import os


  #可视化窗口定义
def visual_window(path,csv_path):
    probe_path=path+'/'+'probe'
    gallery_path=path+'/'+'gallery'
    bgImg = np.zeros((700, 700, 3), np.uint8)#创建一个700*700全黑背景图
    csvfile = open(csv_path, 'r', encoding="utf-8")
    reader = csv.reader(csvfile)
    for row in reader:
        img = cv.imread(probe_path + '/' + str(row[0]))
        new_img = cv.resize(img, (100, 100))
        height, width, c = new_img.shape
        bgImg[0:height, 0:width, :] = new_img
        for n in range(1, 72):
            if n % 2 == 0:
                continue
            name = str(row[n])
            new_path = os.path.join(gallery_path, name)
            img_ = cv.imread(os.path.join(new_path, os.listdir(new_path)[0]))
            new_img_ = cv.resize(img_, (100, 100))
            acc = float(row[n + 1])
            acc = round(acc, 2)  # round函数将W保留两位小数
            text = str(acc)
            cv.putText(new_img_, text, (20, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)#在图片上写入字符
            height, width, c = new_img_.shape
            k = (n // 2) + 1 #图片张数
            new_height = (k // 6) * height
            new_width = (k % 6) * width
            if (k % 6) != 0:
                bgImg[new_height:(height + new_height), new_width:(width + new_width), :] = new_img_
            else:
                bgImg[(new_height - height):new_height, 6 * width:width * 7, :] = new_img_
        cv.namedWindow("Image")
        cv.imshow("Image", bgImg)
        cv.waitKey(0)

#调用函数
path = 'E:\Image_change\Image_change'
csv_path='E:/test.csv'
visual_window(path,csv_path)






















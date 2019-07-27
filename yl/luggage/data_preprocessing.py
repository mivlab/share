import os
import cv2
import shutil
import numpy as np
import math
import random
from DataAugment import img_rotation
from train_val_split import train_val_test_split

def affine_(img, dsize, pi, angle):  #仿射变化
    h,w,c = img.shape
    dist = 1 #原相机到观测平面距离
    #pi = random.randint(0,30)
    pitch = pi*3.14/180.0
    #angle = random.randint(0,360)
    roll = angle * 3.14 / 180.0
    R_AB = np.array([[math.cos(roll), math.sin(roll), 0], [-math.sin(roll), math.cos(roll), 0], [0, 0, 1]]).T
    R_BC = np.array([[1, 0, 0], [0, math.cos(pitch), math.sin(pitch)], [0, -math.sin(pitch), math.cos(pitch)]]).T
    R_AD = np.dot(R_AB, R_BC)
    t_AD = np.array([-math.sin(roll), math.cos(roll), 0]).reshape(3, 1)
    offset = dist * math.tan(pitch)  # 虚拟相机偏离原相机中心距离
    t_AD = t_AD * offset
    K = np.array([[w / 2, 0, w / 2], [0, w / 2, h / 2], [0, 0, 1]])
    dstK = np.array([[dsize[0] / 3, 0, dsize[0] / 2], [0, dsize[0] / 3, dsize[1] / 2], [0, 0, 1]])
    R_DA = np.linalg.inv(R_AD)
    t_DA = -np.dot(R_DA, t_AD)
    n = np.array([0, 0, 1]).reshape(1, 3)
    nd = n / dist
    K_inv = np.linalg.inv(K)
    dot1 = np.dot(t_DA, nd)
    H = np.dot(dstK, np.dot((R_DA + np.dot(t_DA, nd)), K_inv))
    dst_img = cv2.warpPerspective(img, H, dsize)
    # pts = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]]).T
    # pts1 = np.dot(H1, pts)
    # pts1 = pts1 / pts1[2,:]
    # print(pts1)
    # cv2.imshow('image', img)
    # cv2.imshow('dst_image', dst_img)
    # cv2.waitKey()
    return dst_img

def gen_rect(p1, p2, w, h, n):
    """
    从1个矩形框生成n个矩形框，p1，p2为左上角和右下角点，w、h为图像宽高
    Return：
        矩形信息的list
    """
    lmax = int(max(p2[0] - p1[0], p2[1] - p1[1])) # 长边
    lmin = int(min(p2[0] - p1[0], p2[1] - p1[1])) # 短边
    r = lmax // 2 # 矩形中心范围
    cx = (p1[0] + p2[0]) // 2 # 原矩形中心x
    cy = (p1[1] + p2[1]) // 2 # 原矩形中心y
    i = 0
    rects = []
    while i < n:
        cx_ = random.randint(cx - r, cx + r) # 矩形中心坐标x
        cy_ = random.randint(cy - r, cy + r) # 矩形中心坐标y
        width = random.randint(lmin, lmax * 3)  # 矩形宽度
        height = int(width * random.uniform(3.0 / 4, 4.0 / 3))  # 矩形高度（按宽高比范围计算）
        x1 = max(cx_ - width // 2, 0)
        y1 = max(cy_ - height // 2, 0)
        x2 = x1 + width
        y2 = y1 + height
        if x2 >= w or y2 >= h:
            continue
        # 交集与gt的面积之比
        ios = intersection_over_self([p1[0], p1[1], p2[0], p2[1]], [x1, y1, x2, y2])
        # 新矩形是否覆盖gt的一条边
        #cover_side = is_cover_side([p1[0], p1[1], p2[0], p2[1]], [x1, y1, x2, y2])
        #if (cover_side == False and ios < 0.2) or ios < 0.4:
        if ios < 0.35:
            continue
        rects.append([x1, y1, x2, y2])
        i = i + 1
    return rects


def gen_sample(path, dst_path, box):
    '''
    path目录下的图片（含子目录），生成训练样本，并保存到dst_path，以子目录保存
    '''
    for dir in os.listdir(path):
        for name in os.listdir(os.path.join(path, dir)):
            img = cv2.imread(os.path.join(path, dir, name))
            h, w, c = img.shape
            lmax = max(h, w)
            for i in range(10):
                angle = int(random.normalvariate(0, 0.5) * 30)
                rotatedImg = affine_(img, (lmax, lmax), 0, angle)
                image_id = image_name[:4]
                os.makedirs(os.path.join(dst_path, image_id), exist_ok=True)
                rects = gen_rect((x1, y1), (x2, y2), img.shape[1], img.shape[0], num)
                for i, rect in enumerate(rects):
                    img_ = cv2.resize(img[rect[1]:rect[3], rect[0]:rect[2]], (box, box))
                    out_name = os.path.basename(image_name)[0:-4] + '_%d_%d.jpg' % (j, i)
                    cv2.imwrite(os.path.join(save_path, out_name), img_)


if __name__ == '__main__':
    datapath = r'D:\data\xb-reid\MVB_train\Image'
    train_path = r'D:\data\xb-reid\MVB_train\Image_train'
    val_path = r'D:\data\xb-reid\MVB_train\Image_val'
    test_path = r'D:\data\xb-reid\MVB_train\Image_test'
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    train_val_test_split(datapath, train_path, val_path, test_path)

    for image_name in sorted(os.listdir(datapath)):

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


def compute_iou(box1, box2):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """

    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    ## 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou

def intersection_over_self(box1, box2):
    """
    计算两个矩形框box1,box2的交集 / box1的面积.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax]
    Return:
        ios: ios of box1 and box2.
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    ## 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    ios = inter_area / (area1 + 1e-6)
    return ios

def gen_rect(p1, p2, w, h, n):
    """
    从1个矩形框生成n个矩形框，p1，p2为左上角和右下角点，w、h为图像宽高
    Return：
        矩形信息的list
    """
    lmax = int(max(p2[0] - p1[0], p2[1] - p1[1])) # 长边
    lmin = int(min(p2[0] - p1[0], p2[1] - p1[1])) # 短边
    r = lmin // 2 # 矩形中心范围
    cx = (p1[0] + p2[0]) // 2 # 原矩形中心x
    cy = (p1[1] + p2[1]) // 2 # 原矩形中心y
    i = 0
    rects = []
    while i < n:
        cx_ = random.randint(cx - r, cx + r) # 矩形中心坐标x
        cy_ = random.randint(cy - r, cy + r) # 矩形中心坐标y
        width = random.randint(lmin, lmax)  # 矩形宽度
        height = width  # 矩形高度（按宽高比范围计算）
        x1 = max(cx_ - width // 2, 0)
        y1 = max(cy_ - height // 2, 0)
        x2 = x1 + width - 1
        y2 = y1 + height - 1
        if x2 >= w or y2 >= h:
            continue
        # 交集与gt的面积之比
        ios = intersection_over_self([p1[0], p1[1], p2[0], p2[1]], [x1, y1, x2, y2])
        if ios < 0.7:
            continue
        rects.append([x1, y1, x2, y2])
        i = i + 1
    return rects


def get_bbox(img):
    if len(img.shape) == 3:
        img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_ = img
    col_max = np.max(img_, 0) # 每列的最大值
    row_max = np.max(img_, 1) # 每行的最大值
    x1 = 0
    y1 = 0
    x2 = img.shape[1] - 1
    y2 = img.shape[0] - 1
    while row_max[y1] == 0:
        y1 += 1
    while row_max[y2] == 0:
        y2 -= 1
    while col_max[x1] == 0:
        x1 += 1
    while col_max[x2] == 0:
        x2 -= 1
    return [x1, y1, x2, y2]


def gen_sample(path, dst_path, box):
    '''
    path目录下的图片（含子目录），生成训练样本，并保存到dst_path，以子目录保存,
    box: 保存正方形图像的边长
    '''
    for dir in os.listdir(path):
        for k, name in enumerate(os.listdir(os.path.join(path, dir))):
            print(str(k) + ' ' + name)
            img = cv2.imread(os.path.join(path, dir, name))
            h, w, c = img.shape
            lmax = max(h, w)
            for j in range(15):
                base = [0, 90, 180, 270]
                base_i = random.randint(0, 3)
                angle = int(random.normalvariate(0, 0.5) * 30) + base[base_i]
                rotatedImg = affine_(img, (lmax, lmax), 0, angle)
                #cv2.imshow('img', rotatedImg)
                #cv2.imshow('rot', rotatedImg)
                #cv2.waitKey()
                image_id = name[:4]
                os.makedirs(os.path.join(dst_path, image_id), exist_ok=True)
                bbox = get_bbox(rotatedImg)
                rects = gen_rect((bbox[0], bbox[1]), (bbox[2], bbox[3]), rotatedImg.shape[1], rotatedImg.shape[0], 5)
                for i, rect in enumerate(rects):
                    img_ = cv2.resize(rotatedImg[rect[1]:rect[3], rect[0]:rect[2]], (box, box))
                    out_name = os.path.basename(name)[0:-4] + '_%d_%d.jpg' % (angle, i)
                    cv2.imwrite(os.path.join(dst_path, image_id, out_name), img_)


if __name__ == '__main__':
    root = r'C:\data\MVB_train'
    datapath = os.path.join(root, 'Image')
    train_path = os.path.join(root, 'Image_train')
    val_path = os.path.join(root, 'Image_val')
    test_path = os.path.join(root, 'Image_test')
    train_sample = os.path.join(root, 'train_224')
    val_sample = os.path.join(root, 'val_224')

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(train_sample, exist_ok=True)
    os.makedirs(val_sample, exist_ok=True)

    #train_val_test_split(datapath, train_path, val_path, test_path)

    gen_sample(train_path, train_sample, 224)
    gen_sample(val_path, val_sample, 224)

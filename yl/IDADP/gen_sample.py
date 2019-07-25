import os
import cv2
import random
import json
import numpy as np

# 原始图形子目录名
dir_name = [ '1-100', '2-100','3-100', '4-100', '5-100', '6-100']


def is_cover_side(box1, box2):
    """
    计算两个矩形框box2是否覆盖box1的一条边.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax]
    Return:
        True / False.
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    if xmin2 <= xmin1 and xmax2 >= xmax1:
        return True
    elif ymin2 <= ymin1 and ymax2 >= ymax1:
        return True
    else:
        return False

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

def train_val_split(image_root, train_file, val_file):
    """
    划分训练集和验证集.
        image_root: 原始图像根目录
        train_file, val_file: 输出图像列表txt文件，格式为"图像文件名 类别id"
    """
    ftrain = open(train_file, 'w')
    fval = open(val_file, 'w')
    train_names = []
    val_names = []
    for i, dir in enumerate(dir_name):
        names = os.listdir(os.path.join(image_root, dir))
        #random.shuffle(names)
        names = [os.path.join(image_root, dir, name + ' %d\n' % i) for name in names if os.path.splitext(name)[1] == '.JPG']
        num = int(len(names) * 0.85) # train的比例
        train_names.extend(names[0:num])
        val_names.extend(names[num:])
    ftrain.writelines(train_names)
    fval.writelines(val_names)
    ftrain.close()
    fval.close()

def json_get_point(json_name):
    """
    从json文件提取目标点的信息。
    Return：
        点信息的list
    """
    with open(json_name, 'r') as load_f:
        load_dict = json.load(load_f)
    pts_total = []
    for shape in load_dict['shapes']:
        pts_total.append(shape['points'])
    return pts_total

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

def rotate_image(img):
    return img
def rotate_points(pts):
    return pts

def gen_sample(list_file, num, box, target):
    """
    生成训练样本，存到指定目录下。
        list_file: 文件列表，格式为"图像文件名 id"
        num: 每张图生成样本数量
        box：输出正方形图像边长（像素）
        target：输出目录
    """
    f = open(list_file, 'r')
    names = f.readlines()
    for n, name in enumerate(names):
        image_name, c = name.strip().split()
        json_name, ext = os.path.splitext(image_name)
        json_name = json_name + '.json'
        pts_total = json_get_point(json_name)
        img = cv2.imread(image_name)
        if img is None:
            print(image_name + ' does not exist')
            continue
        save_path = os.path.join(target, c)
        os.makedirs(save_path, exist_ok=True)
        print('%d / %d' % (n, len(names)))
        for j, pts in enumerate(pts_total):
            img = rotate_image(img)
            pts = rotate_points(pts)
            pts_ = np.array(pts, dtype=np.int32)
            x1, y1 = np.min(pts_, 0)
            x2, y2 = np.max(pts_, 0)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255))
            rects = gen_rect((x1, y1), (x2, y2), img.shape[1], img.shape[0], num)
            for i, rect in enumerate(rects):
                img_ = cv2.resize(img[rect[1]:rect[3], rect[0]:rect[2]], (box, box))
                out_name = os.path.basename(image_name)[0:-4] + '_%d_%d.jpg' % (j, i)
                cv2.imwrite(os.path.join(save_path, out_name), img_)
        #cv2.imshow('img', img)
        #cv2.waitKey()

if __name__ == '__main__':
    source_root = r'F:\data\competition\datian\IDADP-100' # 原图根目录
    target_train = r'F:\data\competition\datian\IDADP-100-train' # 输出train数据目录
    target_val = r'F:\data\competition\datian\IDADP-100-val' # 输出val数据目录
    os.makedirs(target_train, exist_ok=True)
    os.makedirs(target_val, exist_ok=True)
    train_val_split(source_root, 'train.txt', 'val.txt')
    gen_sample('train.txt', 50, 224, target_train)
    gen_sample('val.txt', 50, 224, target_val)

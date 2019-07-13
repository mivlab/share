import numpy as np
import os
import torch
import cv2
import csv
import time
import argparse
from torch.autograd import Variable
from torchvision import transforms
from model import resnet

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--datapath', required=True, help='valid data path')
parser.add_argument('--checkpoint', default='./output/params_20.pth')
parser.add_argument('--cal_rank', default=True, help='use probe with label to cal rank')
args = parser.parse_args()


# 补方
def fill(img_path, fix=256):
    img = cv2.imread(img_path)
    h, w, c = img.shape
    if h > w:
        re_size = (w * fix // h, fix)
    else:
        re_size = (fix, h * fix // w)
    img_resize = cv2.resize(img, re_size)
    if img_resize.shape[0] == fix:
        img_fix = cv2.copyMakeBorder(img_resize, 0, 0, int((fix - img_resize.shape[1]) / 2),
                                     fix - img_resize.shape[1] - int((fix - img_resize.shape[1]) / 2),
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        img_fix = cv2.copyMakeBorder(img_resize, int((fix - img_resize.shape[0]) / 2),
                                     fix - img_resize.shape[0] - int((fix - img_resize.shape[0]) / 2), 0, 0,
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img_fix


def data_process(path1, path2):
    path1_p = os.path.join(path1, 'probe')
    path1_g = os.path.join(path1, 'gallery')
    path2_p = os.path.join(path2, 'probe')
    path2_g = os.path.join(path2, 'gallery')
    os.makedirs(path2_p, exist_ok=True)
    os.makedirs(path2_g, exist_ok=True)
    for img_name in os.listdir(path1_p):
        img = fill(os.path.join(path1_p, img_name))
        cv2.imwrite(os.path.join(path2_p, img_name), img)
    for img_name in os.listdir(path1_g):
        img_id = img_name[:4]
        if not os.path.exists(os.path.join(path2_g, img_id)):
            os.makedirs(os.path.join(path2_g, img_id))
        img = fill(os.path.join(path1_g, img_name))
        cv2.imwrite(os.path.join(path2_g, img_id, img_name), img)


def rank(label, file, k):
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        for line in reader:
            data.append(line)
    rank_k = np.zeros(len(label))
    for j in range(len(label)):
        if label[j] in data[j][1:2*k:2]:
            rank_k[j] = 1
    return np.mean(rank_k)


def strategy_mean(confidence_array, p_array, g_array_dir, n):
    g_array_one = np.mean(g_array_dir, axis=0)
    for m in range(len(p_array)):
        confidence = np.dot(p_array[m], g_array_one) / (np.linalg.norm(p_array[m]) * (np.linalg.norm(g_array_one)))
        confidence_array[m][n] = np.around(confidence, decimals=3)
    return confidence_array


def strategy_max(confidence_array, p_array, g_array_dir, n):
    for m in range(len(p_array)):
        confidence_max = 0
        for z in range(len(g_array_dir)):
            confidence = np.dot(p_array[m], g_array_dir[z]) / (np.linalg.norm(p_array[m]) * (np.linalg.norm(g_array_dir[z])))
            if confidence > confidence_max:
                confidence_max = confidence
        confidence_array[m][n] = np.around(confidence_max, decimals=3)
    return confidence_array


def get_probe_label(p_name):
    label_array = []
    for i in range(len(p_name)):
        label_array.append(p_name[i][:4])
    return label_array


def test(path):
    model = resnet.resnet18(num_classes=3215)
    model.load_state_dict(torch.load(args.checkpoint))
    model.cuda()
    model.eval()

    # probe
    list_probe = os.listdir(os.path.join(path, 'probe'))
    p_array = np.zeros((len(list_probe), 512))  # m个待查询的probe特征
    p_name = []
    for i, img_p in enumerate(sorted(list_probe)):
        p_name.append(img_p)
        img = cv2.imread(os.path.join(path, 'probe', img_p))
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)
        feature, out = model(Variable(img_tensor.cuda()))
        p_array[i] = feature.cpu().detach().numpy()
        
    # gallery && confidence[m][n]
    list_gallery = os.listdir(os.path.join(path, 'gallery'))
    confidence_array = np.zeros((len(p_array), len(list_gallery)), dtype=np.float)
    for n, dir_name in enumerate(sorted(list_gallery)):  # n个id对应的文件夹
        dir_list = os.listdir(os.path.join(path, 'gallery', dir_name))
        g_array_dir = np.empty((len(dir_list), 512))  # 每个文件夹对应生成的特征
        for i, img_g in enumerate(sorted(dir_list)):
            img = cv2.imread(os.path.join(path, 'gallery', dir_name, img_g))
            img_tensor = transforms.ToTensor()(img).unsqueeze(0)
            feature, out = model(Variable(img_tensor.cuda()))
            g_array_dir[i] = feature.cpu().detach().numpy()
        # confidence_array = strategy_mean(confidence_array, p_array, g_array_dir, n)
        confidence_array = strategy_max(confidence_array, p_array, g_array_dir, n)

    # 将结果写入csv文件中
    with open('output/test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for m in range(confidence_array.shape[0]):
            line = {}
            write_line = [p_name[m]]
            for n in range(confidence_array.shape[1]):
                line[('000' + str(n))[-4:]] = confidence_array[m][n]
            line_reverse = sorted(line.items(), key=lambda x: x[1], reverse=True)
            for t in range(len(line)):
                write_line.append(line_reverse[t][0])
                write_line.append(line_reverse[t][1])
            writer.writerow(write_line)
    return p_name


if __name__ == '__main__':
    start = time.time()
    outpath = os.path.join(args.datapath[:-5], 'Image_change')
    data_process(args.datapath, outpath)
    p_name = test(outpath)
    end = time.time()
    print('Time average: %.3f S' % ((end - start) / len(p_name)))

    if args.cal_rank:
        label_array = get_probe_label(p_name)
        for k in [1, 3, 5, 10]:
            p = rank(label_array, 'output/test.csv', k)
            print('rank%2d : %.3f' % (k, p))

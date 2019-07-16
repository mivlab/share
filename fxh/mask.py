import os
import torchvision.transforms as tfs
from PIL import Image
import random
import numpy as np
import cv2 as cv
import json


path='E:\MVB'
list_path = os.listdir(path)
for dir_name in list_path:#抽取每类一张图作为被遮挡的图，这里的数量可在后面sample处进行调整
    path_list1=os.listdir(os.path.join(path, dir_name))
    #k = int(len(path_list1)* 0.1)
    sample = random.sample(path_list1[:3], 1) #此处sample的第二个值是可以改变的
    for name in sample: #遍历一定数量的被遮挡图
        img1_path = path + '/' + str(dir_name)
        img1 = cv.imread(img1_path+'/'+str(name))
        #cv.imshow("img1",img1)
        #rand_pic_file = str(random.sample(list_path, 1))[2:6]#在除其他类里随机抽取一个类，取其名字
    while True:
        while((str(random.sample(list_path, 1))[2:6])!= str(dir_name)):#判断该类是否与被遮挡图的类相同
            rand_pic_file = str(random.sample(list_path, 1))[2:6]
            path_list2=os.listdir(os.path.join(path, rand_pic_file))
            while True:
                rand_pic=random.sample(path_list2,1)#随机取的一张图
                if str(rand_pic)[2:4]!='H_':
                    for rand_name in rand_pic:
                        img2=cv.imread(path+'/'+str(rand_pic_file)+'/'+str(rand_name))
                        #cv.imshow('img2',img2)
                        height,width=img2.shape[:2]
                        size=(int(width*0.5),int(height*0.5))
                        shrink=cv.resize(img2,size)
                    break
        with open(r'E:\MVB_train\MVB_train\Info\train.json', 'r', encoding='utf-8') as f:  # 读json文件
            for line in f:
                json_dict = json.loads(line)
            for item in json_dict['image']:
                if str(item['image_name']) == str(rand_name):
                    image_name = item['image_name']
                    mask = item['mask']
                   #print('%s:%s' % (image_name, mask))
                    # os.system("pause")
                else:
                    continue
        M = []
        for i in range(0, len(mask), 2):
            M.append(mask[i:i + 2])
        #print(M)
        m = np.array(M)
        m = m // 2  # 进行放缩
        #print(m)
        x2, y2 = np.max(m, 0)
        x1, y1 = np.min(m, 0)
        mask2_ = np.zeros((y2 - y1, x2 - x1, 3), np.uint8)
        img2__ = shrink[y1:y2, x1:x2, :]
        # offset[:,] = [x1, y1]
        m[:, 0] = m[:, 0] - x1
        m[:, 1] = m[:, 1] - y1
        cv.fillConvexPoly(mask2_, m, (255, 255, 255))
        #cv.imshow("mask2_", mask2_)

        #offset1 = random.randint(0, (img1.shape[1] - mask2_.shape[1]))  # 宽
        #offset2 = random.randint(0, (img1.shape[0] - mask2_.shape[0]))  # 高
        if (img1.shape[1] - mask2_.shape[1])>0 and (img1.shape[0] - mask2_.shape[0])>0:
            offset1 = random.randint(0, (img1.shape[1] - mask2_.shape[1]))  # 宽
            offset2 = random.randint(0, (img1.shape[0] - mask2_.shape[0]))
            img1[offset2:offset2 + mask2_.shape[0], offset1:offset1 + mask2_.shape[1], :] = \
            np.where(mask2_ == 0, img1[offset2:offset2 + mask2_.shape[0], offset1:offset1 + mask2_.shape[1], :],img2__)
            cv.imwrite(os.path.join(img1_path, 'H_' + name), img1)
            break


        #cv.imshow('aug', img1)
        #cv.waitKey()
        #cv.imwrite(os.path.join(img1_path, 'H_' + name), img1)

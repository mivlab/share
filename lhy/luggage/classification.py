import os
import shutil
import random

path1 = r'D:\Image_data\train'  #数据路径
path2 = r'D:\Image_data\val'    #生成验证集路径

list_path1 = os.listdir(path1)
for dir_name in list_path1:
    file = os.listdir(os.path.join(path1, dir_name))
    k = len(file)  #每个文件夹内图片个数
    move = int(k * 0.2)  #需要移动的个数
    if move > 0 and (not os.path.exists(os.path.join(path2, dir_name))):
        os.makedirs(os.path.join(path2, dir_name))
    for j in range(move):
        x = random.randint(0, (len(file)-1))
        shutil.move(os.path.join(path1, dir_name, file[x]),os.path.join(path2, dir_name, file[x]))
        file.remove(file[x])

import os
import shutil
import random

path1 = r'D:\newdata\train'  #数据路径
path2 = r'D:\newdata\val'    #生成验证集路径

list_path1 = os.listdir(path1)
for dir_name in list_path1:
    k = len(os.listdir(os.path.join(path1, dir_name)))
    file = os.listdir(os.path.join(path1, dir_name))
    for j in range(k):
        if j==k-1 and len(os.path.join(path1, dir_name))==1:  #保证每个文件见至少还存在一张图片  20%概率抽出至验证集
            continue
        else:
            x = random.randint(0,100)
            if x < 20:
                if not os.path.exists(os.path.join(path2, dir_name)):
                    os.makedirs(os.path.join(path2, dir_name))
                shutil.move(os.path.join(path1, dir_name, file[j]),os.path.join(path2, dir_name, file[j]))

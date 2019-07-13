import os
import torchvision.transforms as tfs
from PIL import Image
import random
import numpy as np

datapath = r'D:\tupian'

for dir in sorted(os.listdir(datapath)):
    #print(dir)
    path = os.path.join(datapath, dir)
    # 水平垂直翻转
    for name in os.listdir(path):
        img = Image.open(os.path.join(path, name))
        img_flip_h = tfs.RandomHorizontalFlip(p=1)(img)
        img_flip_v = tfs.RandomVerticalFlip(p=1)(img)
        img_flip_h.save(os.path.join(path, 'F0_' + name))
        img_flip_v.save(os.path.join(path, 'F1_' + name))


    # 中心裁剪
    for name in os.listdir(path):
        img = Image.open(os.path.join(path, name))
        size_scale = (int(img.size[1] * 1.2), int(img.size[0] * 1.2))
        img_resize = tfs.Resize(size_scale, interpolation=2)(img)
        img_crop = tfs.RandomCrop((img.size[1], img.size[0]), padding=0, pad_if_needed=False)(img_resize)
        img_crop.save(os.path.join(path, 'C0_' + name))

    # 旋转
    for name in os.listdir(path):
        img = Image.open(os.path.join(path, name))
        img_rot_1 = tfs.RandomRotation(30, resample=False, expand=False, center=None)(img)
        img_rot_2 = tfs.RandomRotation(30, resample=False, expand=False, center=None)(img)
        img_rot_1.save(os.path.join(path, 'R0_' + name))
        img_rot_2.save(os.path.join(path, 'R1_' + name))

    #   亮度
    for name in os.listdir(path):
        img = Image.open(os.path.join(path, name))
        img_clj_1 = tfs.ColorJitter(brightness=0.8, contrast=0, saturation=0, hue=0)(img)
        img_clj_1.save(os.path.join(path, 'J0_' + name))


#遮挡
x=random.randint(0,len(os.listdir(datapath))-1)
dir_name=('0000'+ str(x))[-4 :]
path=os.path.join(datapath,dir_name)
img_list=os.listdir(path)
y=random.randint(0,len(img_list)-1)
img_=Image.open(os.path.join(path,img_list[y]))
img_crop = tfs.CenterCrop((256*0.8, 256*0.8))(img_)
img_resize =img_crop.resize((64,64),Image.ANTIALIAS)


for dir in os.listdir(datapath):
    path1 = os.path.join(datapath, dir)
    for name in os.listdir(path1):
        x=random.randint(0,100)
        if x < 20:
            img = np.array(Image.open((os.path.join(path1, name))),dtype=np.uint8)
            offset = random.randint(0,192)
            img[offset:offset+64, offset:offset+64, :] = img_resize
            im = Image.fromarray(img)
        else:
            continue

        im.save(os.path.join(path1,'H0_' + name))
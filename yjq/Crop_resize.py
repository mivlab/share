import os
import torchvision.transforms as tfs
import numpy as np
import random
from PIL import Image


datapath = r'D:\t\test'
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
        if x < 5:
            img = np.array(Image.open((os.path.join(path1, name))),dtype=np.uint8)
            offset = random.randint(0,192)
            img[offset:offset+64, offset:offset+64, :] = img_resize
            im = Image.fromarray(img)
        else:
            continue

        im.save(os.path.join(path1,'c_2' + name))
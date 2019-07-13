import os
import torchvision.transforms as tfs
from PIL import Image

datapath = r'/home/lhy/newdata/train/'

for dir in sorted(os.listdir(datapath)):
    print(dir)
    path = os.path.join(datapath, dir)
    # Flip Fx_
    for name in os.listdir(path):
        img = Image.open(os.path.join(path, name))
        img_flip_h = tfs.RandomHorizontalFlip(p=1)(img)
        img_flip_v = tfs.RandomVerticalFlip(p=1)(img)
        img_flip_h.save(os.path.join(path, 'F0_' + name))
        img_flip_v.save(os.path.join(path, 'F1_' + name))


    # Crop C_
    for name in os.listdir(path):
        img = Image.open(os.path.join(path, name))
        size_scale = (int(img.size[1] * 1.2), int(img.size[0] * 1.2))
        img_resize = tfs.Resize(size_scale, interpolation=2)(img)
        img_crop = tfs.RandomCrop((img.size[1], img.size[0]), padding=0, pad_if_needed=False)(img_resize)
        img_crop.save(os.path.join(path, 'C_' + name))

    # Rotation Rx_
    for name in os.listdir(path):
        img = Image.open(os.path.join(path, name))
        img_rot_1 = tfs.RandomRotation(30, resample=False, expand=False, center=None)(img)
        img_rot_2 = tfs.RandomRotation(30, resample=False, expand=False, center=None)(img)
        img_rot_1.save(os.path.join(path, 'R0_' + name))
        img_rot_2.save(os.path.join(path, 'R1_' + name))

    #   ColorJitter Jx_
    for name in os.listdir(path):
        img = Image.open(os.path.join(path, name))
        img_clj_1 = tfs.ColorJitter(brightness=0.8, contrast=0, saturation=0, hue=0)(img)
        img_clj_1.save(os.path.join(path, 'J0_' + name))
from PIL import Image
import os

path = r'/home/data/wxy/IDADP-100/test'
for dir_name in os.listdir(path):
    file = os.path.join(path, dir_name)
    for pic_name in os.listdir(file):
        im = Image.open(os.path.join(file, pic_name))
        img_size = im.size
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0
        for n in range(13):
            for m in range(21):
                w = 500
                h = 500
                x1 = 125 * m
                x2 = 125 * m + w
                y1 = n * 125
                y2 = y1 + h
                img1 = im.crop((x1, y1, x2, y2))
                img_ = img1.resize((224, 224))
                out_name = (os.path.join(file, pic_name)[:-4] + '_%d_%d.jpg' % (n, m))
                path2 = r'/home/wxy/IDADP-100/IDADP-100-test'
                if not os.path.exists(os.path.join(path2, dir_name)):
                    os.mkdir(os.path.join(path2, dir_name))
                img_.save(os.path.join(path2, dir_name, out_name[32:]))


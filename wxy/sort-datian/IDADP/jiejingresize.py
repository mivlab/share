import os
import cv2

path = r'/home/wxy/IDADP-100/IDADP-100-train'
for dic_name in os.listdir(path):
    file = os.path.join(path, dic_name)
    for img in os.listdir(file):
        img_ = cv2.imread(os.path.join(file, img))
        img__ = cv2.resize(img_, (224, 224))
        cv2.imwrite(os.path.join(path,dic_name, img), img__)

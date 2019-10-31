import torch
import torch.nn.functional as f
import cv2
from torch.autograd import Variable
from torchvision import transforms, models
from PIL import Image
from model.resnet import resnet18
import os
import shutil
import numpy as np
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = models.resnet18(num_classes=7)
# model.load_state_dict(torch.load('output/params_30.pth'))
# model = torch.load('output/model.pth')
model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('output/params_ending.pth').items()})
model.eval()
if torch.cuda.is_available():
    model.cuda()
path = r'/home/wxy/IDADP-result-test/danyi100'
path1 = r'/home/wxy/IDADP-PRCV2019-test/danyi100'
path2 = r'/home/wxy/danyi'
# path = r'/home/wxy/IDADP-result-test'
# path1 = r'/home/wxy/IDADP-100/IDADP-100-obersavation'
# for id, dir_name in enumerate(sorted(os.listdir(path))):
#     wxy = np.zeros((6,))
#     file = os.path.join(path, dir_name)

for k, pic_name in enumerate(sorted(os.listdir(path))):
    if k % 273 == 0:
        xy = np.zeros((7,))
    pic_name1 = os.path.join(path, pic_name)
    img = cv2.imread(pic_name1)
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        prediction = model(Variable(img_tensor.cuda()))
    else:
        prediction = model(Variable(img_tensor))
    prediction = f.softmax(prediction)
    pred = torch.max(prediction, 1)[1].item()
    if torch.max(prediction, 1)[0].item() >= 0.90:
        xy[pred] = xy[pred] + 1
        # if not os.path.exists(os.path.join(path1, dir_name)):
        #     os.mkdir(os.path.join(path1, dir_name))
        # if torch.max(prediction, 1)[1].item() != id:
        #     cv2.imwrite(os.path.join(path1, dir_name, pic_name), img)

    if (k + 1) % 273 == 0:
        y = int(np.argmax(xy[:-1]))
        if not os.path.exists(os.path.join(path2, str(y))):
            os.mkdir(os.path.join(path2, str(y)))
        shutil.copy(os.path.join(path1, pic_name.split('_', 1)[0] + '.JPG'), os.path.join(path2, str(y)))

        # acc = wxy[id] / sum(wxy)
        # print("class %d  pic_name: %s dir_name: %s " % (y, pic_name))
# print(wxy)
# cv2.imshow("image", img)
# cv2.waitKey(0)

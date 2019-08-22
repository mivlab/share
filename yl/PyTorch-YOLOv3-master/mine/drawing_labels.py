
import json
import cv2
import os
import numpy as np

root = r'F:\data\mine\web'
show_scale = 1
for file in os.listdir(root):
    filename, ext = os.path.splitext(file)
    if ext != '.json':
        continue
    print(file)
    with open(os.path.join(root, file),'r') as load_f:
        load_dict = json.load(load_f)
    #out = open('1.txt', 'w')
    img = cv2.imread(os.path.join(root, load_dict["imagePath"]))
    if img is None:
        print('image file does not exist')
    smallImg = cv2.resize(img, (int(img.shape[1]/ show_scale), int(img.shape[0]/ show_scale)))
    for shape in load_dict['shapes']:
        pts = np.array(shape['points'])
        x2, y2 = np.max(pts, 0) + 1
        x1, y1 = np.min(pts, 0) - 1

        intpt1 = (int(int(x1) / show_scale), int(int(y1) / show_scale))
        intpt2 = (int(int(x2) / show_scale), int(int(y2) / show_scale))
        cv2.rectangle(smallImg, intpt1, intpt2, (0, 0, 255))
        #cv2.circle(smallImg, intpt1, 3, (0, 0, 255))
        #cv2.circle(smallImg, intpt2, 3, (0, 0, 255))
        #out.write('%s %s' % (pts[0], pts[1]))
    cv2.imshow('image', smallImg)
    cv2.waitKey()
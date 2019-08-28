
import json
import cv2
import os
import numpy as np
import ellipses
import random
import config

#原图resize，使目标为指定大小，并保存矩形框信息
if __name__ == '__main__':
    root = config.root
    outDir = config.fgDir
    fixsize = config.fixsize # 目标大小范围
    trainSize = config.trainSize #输出的训练图像大小

    os.makedirs(outDir, exist_ok=True)

    show_scale = 1
    for file in os.listdir(root):
        filename, ext = os.path.splitext(file)
        if ext != '.json':
            continue
        print(file)
        with open(os.path.join(root, file),'r') as load_f:
            load_dict = json.load(load_f)
        img = cv2.imread(os.path.join(root, load_dict["imagePath"]))
        if img is None:
            print('image file does not exist')
        ih, iw, ic = img.shape
        smallImg = cv2.resize(img, (int(img.shape[1]/ show_scale), int(img.shape[0]/ show_scale)))
        objectNum = len(load_dict['shapes'])
        for i, shape in enumerate(load_dict['shapes']):
            pts = np.array(shape['points'])

            if pts.shape[0] >= 6: # circle
                n = pts.shape[0]
                data = [pts[0:n-1, 0], pts[0:n-1, 1]]
                lsqe = ellipses.LSqEllipse()
                lsqe.fit(data)
                center, width, height, phi = lsqe.parameters()
                cv2.ellipse(smallImg, (int(center[0]), int(center[1])), (int(width), int(height)), phi / 3.14159 * 180,
                            0, 360, (255, 0, 0), 2)
                ellpts = cv2.ellipse2Poly((int(center[0]), int(center[1])), (int(width), int(height)), int(phi / 3.14159 * 180),
                            0, 360, 5)
                ellpts = np.append(ellpts, [[pts[n-1, 0], pts[n-1, 1]]], axis=0)
                x2, y2 = np.max(ellpts, 0)
                x1, y1 = np.min(ellpts, 0)

            else: # rectangle
                x2, y2 = np.max(pts, 0)
                x1, y1 = np.min(pts, 0)

            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, iw - 1)
            y2 = min(y2, ih - 1)

            intpt1 = (int(int(x1) / show_scale), int(int(y1) / show_scale))
            intpt2 = (int(int(x2) / show_scale), int(int(y2) / show_scale))
            cv2.rectangle(smallImg, intpt1, intpt2, (0, 0, 255))

            size1 = x2 - x1 + 1
            size = random.randint(fixsize[0], fixsize[1])
            r = size / size1
            outImg = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
            x1 = int(x1 * r)
            x2 = int(x2 * r)
            y1 = int(y1 * r)
            y2 = int(y2 * r)
            name_ = os.path.join(outDir, '%s_%d.jpg' % (filename, i))
            outTxt = open(os.path.join(outDir, '%s_%d.txt' % (filename, i)), 'w')
            # 只有一个物体，并且图像较小，存整张图
            if objectNum == 1 and outImg.shape[0] < trainSize / 2 and outImg.shape[1] < trainSize / 2:
                outTxt.write('0 %f %f %f %f\n' % (x1, y1, x2, y2))
                cv2.imwrite(name_, outImg)
            else: # 有多个物体，或图像较大，截小之后分别保存
                m = random.randint(2, 6) #不能太大，会把其他目标包含进来
                x11 = max(x1 - m, 0)
                x22 = min(x2 + m, outImg.shape[1] - 1)
                y11 = max(y1 - m, 0)
                y22 = min(y2 + m, outImg.shape[0] - 1)
                cv2.imwrite(name_, outImg[y11:y22+1, x11:x22+1, :])
                outTxt.write('0 %f %f %f %f\n' % (x1 - x11, y1 - y11, x2 - x11, y2 - y11))
            outTxt.close()
            #out.write('0 %f %f %f %f\n' % ((x1 + x2) * 0.5 / iw, (y1 + y2) * 0.5 / ih, (x2 - x1 + 1) / iw, (y2 - y1 + 1) / ih))
        cv2.imshow('image', smallImg)
        cv2.waitKey(1)
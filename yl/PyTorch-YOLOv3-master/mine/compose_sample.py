
import cv2
import os
import numpy as np


if __name__ == '__main__':
    fgDir = r'F:\data\mine\web_fixsize' #前景图
    bgDir = r'F:\data\uav\neg' # 背景图目录
    trainImgDir = r'F:\data\mine\train' # 训练图
    trainLabelDir = r'F:\data\mine\train_label' # 训练标签
    trainSize = 416 #输出的训练图像大小

    os.makedirs(trainImgDir, exist_ok=True)
    os.makedirs(trainLabelDir, exist_ok=True)

    fgFiles = []
    for file in os.listdir(fgDir):
        _, ext = os.path.splitext(file)
        if ext == '.jpg':
            fgFiles.append(file)

    fgIndex = 0
    for file in os.listdir(bgDir):
        print(file)
        img = cv2.imread(os.path.join(bgDir, file))
        if img is None:
            print('%s does not exist' % file)
            continue
        h, w, c = img.shape
        step = (trainSize * 3) // 4
        # 遍历背景图
        for row in range(0, h - trainSize, step):
            for col in range(0, w - trainSize, step):
                bgImg = img[row:row + trainSize, col:col + trainSize, :]
                fgImg = cv2.imread(os.path.join(fgDir, fgFiles[fgIndex]))
                if bgImg.shape[0] - fgImg.shape[0] < 0 or bgImg.shape[1] - fgImg.shape[1] < 0:
                    print('%s is too large' % fgFiles[fgIndex])
                    continue
                y1 = (bgImg.shape[0] - fgImg.shape[0]) // 2
                x1 = (bgImg.shape[1] - fgImg.shape[1]) // 2
                bgImg[y1:y1+fgImg.shape[0], x1:x1+fgImg.shape[1], :] = fgImg # 两图叠加
                trainImgName = '%s_%d_%d_%s' %(file, row, col, fgFiles[fgIndex])
                name = os.path.join(trainImgDir, trainImgName)
                cv2.imwrite(name, bgImg)

                filename, ext = os.path.splitext(fgFiles[fgIndex])
                flabel = open(os.path.join(fgDir, filename + '.txt'), 'r')
                annot = flabel.readline().strip().split(' ')
                flabel.close()
                outLableName = os.path.join(trainLabelDir, os.path.splitext(trainImgName)[0] + '.txt')
                outFile = open(outLableName, 'w')
                cx = (float(annot[1]) + float(annot[3])) / 2 + x1
                cy = (float(annot[2]) + float(annot[4])) / 2 + y1
                width = float(annot[3]) - float(annot[1]) + 1
                height = float(annot[4]) - float(annot[2]) + 1
                outFile.write('0 %f %f %f %f\n' % (cx / bgImg.shape[1], cy / bgImg.shape[0], width / bgImg.shape[1], height / bgImg.shape[0]))
                fgIndex = (fgIndex + 1) % len(fgFiles)
                outFile.close()

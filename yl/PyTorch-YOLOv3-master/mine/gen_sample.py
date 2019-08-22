
import json
import cv2
import os
import numpy as np
import ellipses

if __name__ == '__main__':
    root = r'F:\data\mine\web'
    outDir = r'G:\proj\retrieval\share\yl\PyTorch-YOLOv3-master\data\custom\labels'
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
        out = open(os.path.join(outDir, '%s.txt' % filename), 'w')
        for shape in load_dict['shapes']:
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
            out.write('0 %f %f %f %f\n' % ((x1 + x2) * 0.5 / iw, (y1 + y2) * 0.5 / ih, (x2 - x1 + 1) / iw, (y2 - y1 + 1) / ih))
        out.close()
        cv2.imshow('image', smallImg)
        cv2.waitKey(1)
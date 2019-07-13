import os
import cv2
import shutil

datapath = r'D:\workplace\data\MVB_train\Image'
outpath = r'D:\workplace\data\MVB_data'

os.makedirs(os.path.join(outpath, 'train'))
os.makedirs(os.path.join(outpath, 'test', 'probe'))
os.makedirs(os.path.join(outpath, 'test', 'gallery'))

for image in sorted(os.listdir(datapath)):
    # fix 256
    img = cv2.imread(os.path.join(datapath, image))
    h, w, c = img.shape
    if h == max(h, w):
        re_size = (int(256.0/h*w), 256)
    else:
        re_size = (256, int(256.0/w*h))
    img_resize = cv2.resize(img, re_size)
    if img_resize.shape[0] == 256:
        img_fix = cv2.copyMakeBorder(img_resize, 0, 0, int((256-img_resize.shape[1])/2),
                                     256-img_resize.shape[1]-int((256-img_resize.shape[1])/2),
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        img_fix = cv2.copyMakeBorder(img_resize, int((256-img_resize.shape[0])/2),
                                     256-img_resize.shape[0]-int((256-img_resize.shape[0])/2), 0, 0,
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
    cv2.imwrite(os.path.join(datapath, image), img_fix)

    # copy
    image_id = image[:4]
    if int(image_id) < (4019-100):
        rm_path_train = os.path.join(outpath, 'train', str(image_id))
        if not os.path.exists(rm_path_train):
            os.makedirs(rm_path_train)
        shutil.copy(os.path.join(datapath, image), os.path.join(rm_path_train, image))
    elif 'p' in image[:-4]:
        shutil.copy(os.path.join(datapath, image), os.path.join(outpath, 'test', 'probe'))
    else:
        rm_path_test = os.path.join(outpath, 'test', 'gallery', str(image_id))
        if not os.path.exists(rm_path_test):
            os.makedirs(rm_path_test)
        shutil.copy(os.path.join(datapath, image), os.path.join(rm_path_test, image))

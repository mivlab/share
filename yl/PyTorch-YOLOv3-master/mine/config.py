root = r'F:\data\mine\web'  # 含目标的原图
fgDir = r'F:\data\mine\web_fixsize'  # 前景图目录
bgDir = r'F:\data\uav\neg'  # 背景图目录
trainImgDir = r'F:\data\mine\images'  # 训练图. yolov3程序要求目录名为images和labels
trainLabelDir = r'F:\data\mine\labels'  # 训练标签
fixsize = [16, 32]  # 目标大小范围
trainSize = 416  # 输出的训练图像大小
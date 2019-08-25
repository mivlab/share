gen_sample.py  由标注的6个点生成矩形框，并写入txt。每个jpg对应一个同名的txt，每个txt一行代表一个矩形框。
gen_train_list.py  把目录下的图像分为train和valid，分别存为图像列表。


yolo v3训练步骤：
1. 执行gen_sample.py
2. 执行gen_train_list.py
3. 执行根目录下train.py，开始训练。

小目标检测训练步骤：
0. 修改config.py中的目录
1. gen_sample_fixsize.py 对每张图，由标注的6个点生成矩形框，对原图进行缩放，使矩形框变为期望的目标大小。
2. compose_sample.py 从背景图截416x416大小，原图叠加到背景图。保存叠加图的矩形框信息。
3. gen_train_list.py 把图片按比例分为train 和valid两部分。
4. 上一级目录 train.py 开始训练
（note：重点是把原图和背景图都用到，如何组合并不重要。
背景图要用滑动窗口截取，以便穷尽。
train和valid划分并不独立，这部分有待完善。）

测试步骤：
1. detect_all.py 对指定目录下的所有图像进行检测，无论小图或大图



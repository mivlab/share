import os

root = r'G:\proj\retrieval\share\yl\PyTorch-YOLOv3-master\data\custom\images'
path_prefix = r'data\custom\images'

train_txt = open('../data/custom/train.txt', 'w')
val_txt = open('../data/custom/valid.txt', 'w')

files = os.listdir(root)
for i, file in enumerate(files):
    if i < len(files) * 0.8:
        train_txt.write('%s\n' % os.path.join(path_prefix, file))
    else:
        val_txt.write('%s\n' % os.path.join(path_prefix, file))
train_txt.close()
val_txt.close()
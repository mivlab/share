import os

root = r'F:\data\mine\images'
path_prefix = root # r'data\custom\images'

train_txt = open('../data/custom/train.txt', 'w')
val_txt = open('../data/custom/valid.txt', 'w')

files = os.listdir(root)
for i, file in enumerate(files):
    if i < len(files) * 0.9:
        train_txt.write('%s\n' % os.path.join(path_prefix, file))
    else:
        val_txt.write('%s\n' % os.path.join(path_prefix, file))
train_txt.close()
val_txt.close()
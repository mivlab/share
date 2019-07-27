import os
import shutil
import random

def train_val_test_split(path, train_path, val_path, test_path):

    # 复制到train_path下名为id的子目录
    for image_name in os.listdir(path):
        image_id = image_name[:4]
        os.makedirs(os.path.join(train_path, image_id), exist_ok=True)
        shutil.copy(os.path.join(path, image_name), os.path.join(train_path, image_id))

    # 移动一部分到test
    test_list = sorted(os.listdir(train_path))[-100:]
    for test in test_list:
        #os.makedirs(os.path.join(test_path, test), exist_ok=True)
        shutil.move(os.path.join(train_path, test), test_path)

    # 移动一部分到val
    for dir in os.listdir(train_path):
        file = os.listdir(os.path.join(train_path, dir))
        move = int(len(file) * 0.2) # 验证集比例
        if move > 0 and (not os.path.exists(os.path.join(val_path, dir))):
            os.makedirs(os.path.join(val_path, dir))
        for j in range(move):
            x = random.randint(0, (len(file)-1)) # 随机移动一个文件
            shutil.move(os.path.join(train_path, dir, file[x]),os.path.join(val_path, dir, file[x]))
            file.remove(file[x]) # 移走之后把该文件名从列表删除，防止重复移动

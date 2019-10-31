import os
datapath = r'/home/wxy/IDADP-100/IDADP-100-val'
file = open('/home/wxy/MINIST2/mnist_pytorch-master/output/val.txt', 'w')
for dir in sorted(os.listdir(datapath)):
    path = (os.path.join(datapath, dir))
    for name in os.listdir(path):
        a = path[-1:]
        file.write(os.path.join(path, name)+"  " + str(a) + "\n")
        file.flush()
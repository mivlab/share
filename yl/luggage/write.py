import os
datapath = r'D:\tupian\val'
file = open('val.txt', 'w')
for dir in sorted(os.listdir(datapath)):
    path = (os.path.join(datapath, dir))
    for name in os.listdir(path):
        a = int(path[-4:])
        file.write (os.path.join(path, name)+"  "+str(a)+"\n")
        file.flush()
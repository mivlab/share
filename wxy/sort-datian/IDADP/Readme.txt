1.熟悉六类大田的病害特征（basis）
2.数据划分：将每类的图片进行数据划分，将每类的20%的图片作为测试集（Test），其他80%图片作为训练集和验证集。
3.将这80%的图片进行标注，再次数据划分，80%中的15%的图片作为验证集和85%的图片作为验证集。（数据在服务器101：home/data/wxy/IDADP.zip[其中包括标注点和已分测试集和测试集]）

#Train
1.预处理参考代码 gen_sample.py
修改gen_sample.py 代码最后 source_root， target_train， target_val 目录名，即可运行。
2.利用write脚本，将train和val文件夹分别写入train.txt文件和val.txt文件，将这两个文件放入output中（其实gen_sample.py里面也写了train.txt和val.txt,但是注意的是这两个txt写入的是没有预处理过的图片名字，不是我们所需要的。本人无能，不会在源代码上改，if you can，have a try）
3.进行main.py，调整训练数量，数据路径，参数（这里用的是手写字符的.）
4.将生成的.pth模型文件导入test中的output进行测试程序
ps：另外jiejingresize.py是将图片resize成224*224的，如果你发现图片大小不一致的话，可以派上用场。
        不管是运行哪个脚本，数据路径一定得看仔细了

#Test
1.利用slidewindow3.py脚本进行滑窗数据处理。
2.利用test.py进行测试，注意修改路径啥的（这个代码可能有点混乱，改自从手写字符的test）

测试程序
*必须在有cuda的环境下运行
test.py: 需要运行的测试程序，需要传递的3个参数：
          --datapath 验证集的路径，也就是MVB_val中Image的绝对路劲，不能含中文
          --checkpoint 训练好的 .pth文件的路径
          --cal_rank  默认为True，因为从训练集拿部分作为验证，所以含标签可计算rank值。若使用官方的验证集，要改为False（我们这个竞赛没啥要求，不用改）

代码第97行 model = resnet.resnet18(num_classes=      ) 之后要使用自己训练好的模型的化，需要改成自己训练的类别数

pss：虽然在我自己做测试时的效果还可以，但是拿官方数据来测试时出现了不太理想的结果
         如果你有什么想法可以优化，可以探讨一下。
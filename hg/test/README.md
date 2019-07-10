必须在有cuda的环境下运行

model文件：该测试程序是用resnet18模型，如果变动自己修改模型部分
MVB_val: 从测试图片中选取的部分，完全仿照将来要给的验证数据，编号必须从 0000 开始
output： 需要放入之前训练好的模型，运行程序后会产生可提交的 csv文件
test.py: 需要运行的测试程序，需要传递的3个参数：
          --datapath 验证集的路径，也就是MVB_val中Image的绝对路劲，不能含中文
          --checkpoint 训练好的 .pth文件的路径
          --cal_rank  默认为True，因为从训练集拿部分作为验证，所以含标签可计算rank值。若使用官方的验证集，要改为False

代码第97行 model = resnet.resnet18(num_classes=3215) 之后要使用自己训练好的模型的化，需要改成自己训练的类别数
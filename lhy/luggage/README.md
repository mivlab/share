1.先进行preprocessing—预处理，将图片补方并按图片名字分为4019个文件夹
2.进行classfication处理，抽取80%作为训练集，20%作为验证集，训练集仍要保留4019类，为保证数据平衡，每类最多取1张或不取
3.进行数据增强
4.利用write脚本，将train和val文件夹分别写入train.txt文件和train.txt文件，再合并生成total.txt文件，将三个文件放入output中
5.进行main.py，调整训练数量，数据路径，参数
6.将生成的.pth模型文件导入test中的output进行测试程序


测试程序
*必须在有cuda的环境下运行

MVB_val: 从测试图片中选取的部分，与将来要给的验证数据格式一样，编号从 0000 开始
output： 创建一个output文件夹，需要放入之前训练好的.pth模型，运行程序后会产生 csv文件
test.py: 需要运行的测试程序，需要传递的3个参数：
          --datapath 验证集的路径，也就是MVB_val中Image的绝对路劲，不能含中文
          --checkpoint 训练好的 .pth文件的路径
          --cal_rank  默认为True，因为从训练集拿部分作为验证，所以含标签可计算rank值。若使用官方的验证集，要改为False

代码第97行 model = resnet.resnet18(num_classes=3215) 之后要使用自己训练好的模型的化，需要改成自己训练的类别数
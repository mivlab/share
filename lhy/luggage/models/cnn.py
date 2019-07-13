import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(  #3x28x28
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1, #if stride=1,padding=(kernel_size-1)/2
            ),  # 32x28x28  卷积层（过滤器）
            nn.ReLU(),  #激励函数
            nn.MaxPool2d(kernel_size=2), #池化层
        )  # 32x14x14
        self.conv2 = nn.Sequential(   #32x14x14
            nn.Conv2d(32, 64, 3, 1, 1),  # 64x14x14
            nn.ReLU(), #64x14x14
            nn.MaxPool2d(2)  # 64x7x7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # 64x7x7
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x3x3
        )

        self.dense = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),  # fc4 64*3*3 -> 128  全连接层
            nn.ReLU(),
            nn.Linear(128, 10)  # fc5 128->10
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)  # 64x3x3
        res = conv3_out.view(conv3_out.size(0), -1)  # batch x (64*3*3)
        out = self.dense(res)
        return out

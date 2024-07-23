# 0.导包
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 1.定义一个简单的网络类
class Net(nn.Module):
    def __init__(self):
        """
        初始化函数
        """
        super(Net, self).__init__()
        # 定义第一层卷积神经网络，输入通道维度=1，输出通道维度=6，卷积核大小3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 定义第二层卷积神经网络，输入通道维度=6，输出通道维度=16，卷积核大小3*3
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 定义第三层全连接网络
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 16*6*6为输入维度，120为最后神经元的维度
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10是因为最后有10分类

    def forward(self, x):
        """
        前向传播函数
        :param x:
        :return:
        注意：任意卷积层最后要加激活层，池化层
        """
        # 在（2，2）的池化窗口下执行最大池化操作
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # self.conv1()激活层，再经历2*2的池化层
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 经过卷积层的处理后，张量要进入全连接层，进入先要调整张量的形状
        x = x.view(-1, self.num_float_features(x)) # 重新将三维的张量设置为二维
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        # pass

    def num_float_features(self, x):
        # 将三维张量的后两维设置为同一个维度
        # 计算size，除了第0个维度上的batch_size
        size = x.size()[1:] # x正常应该是三维的张量，我们取出后两维
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        # pass


if __name__ == "__main__":
    net = Net()
    print(net)
    input = torch.randn(1, 1, 32, 32)
    output = net(input)
    print("input:\n",input)
    print("output:\n",output)

    # 1.更新网络参数
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()
    # 损失函数
    target = torch.randn(10)
    target = target.view(1, -1) # 改变target的形状为二维张量，为了和output形状匹配
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print("loss:\n", loss)
    loss.backward()
    optimizer.step()

    # 2.反向传播
    net.zero_grad()

    print("conv1.bias.grad before backward")
    print(net.conv1.bias.grad)

    print("conv1.bias.grad after backward")
    print(net.conv1.bias.grad)

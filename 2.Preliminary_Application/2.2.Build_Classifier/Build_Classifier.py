# 0.导包
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1.下载数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True,num_workers=0)
testset = torchvision.datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
classes = ('plane','car','bird','cat','deer','dog','forg','horse','ship','truck')

# 2.定义卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义两个全卷积层
        # 采用三通道
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义池化层
        self.pool = nn.MaxPool2d(2,2)
        # 定义三个全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)   # 这里的84，64自己选择
        self.fc3 = nn.Linear(84, 10)    # 因为有10个类别，所以选择输出特征为10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 变换x的形状，以适配全连接层的输入
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# 3.定义损失函数
net = Net()
criterizer = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 4.在训练集上训练模型
for epoch in range(2):
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        # data中包含输出图像张量input，标签张良label
        input, label = data
        # 首先将优化器梯度归零
        optimizer.zero_grad()
        # 输入图像张量进网络，得到输出张量output
        outputs = net(input)
        # 利用网络的输出output和标签label进行损失计算
        loss = criterizer(outputs, label)
        # 反向传播+参数更新
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 2000 == 0:  # 有2000个批次，每隔2000个批次打印一次
            print("[%d, %5d] loss: % .3f" %
                  (epoch + 1, i + 1, running_loss / 2000))  # epoch，i都是从0开始的，执行+1操作从1开始，loss每2000为一个批次
            running_loss = 0.0
    print("Finished Training")

# 模型保存
# 首先定义模型的保存路径
PATH = './cifar_net.pth'
# 保存模型的状态字典
torch.save(net.state_dict(), PATH)

# 5.在测试集上测试模型
# 5.1.展示测试集中的若干图片
dataiter = iter(testloader) # 构建一个迭代器
images, labels = next(dataiter)
# 打印原始图片
# imshow(torchvision.utils.make_grid(images))
# 打印真实标签
print("GroundTruth:", " ".join('%5s' % classes[labels[j]] for j in range(4)))

# 5.2.加载模型并对模型进行预测
# 首先实例化模型的对象
net = Net()
# 加载训练阶段保存的模型状态字典
net.load_state_dict(torch.load(PATH))

# 为了更加细致的看一下模型在哪些类别上表现得更好，在哪些些类别上表现更差，我们分类别的进行准确率计算
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s: %2d %%' %
          (classes[i], 100 * class_correct[i] / class_total[i]))

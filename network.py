import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.datasets as dsets

# 超参数
EPOCH = 1  # 为了节省时间，只训练一趟
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = True

# MNIST数字数据集
train_data = dsets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

# 打印出其中一个例子
print(train_data.train_data.size())  # (60000, 28, 28)
print(train_data.train_labels.size())  # (60000)
plt.imshow(train_data.train_data[1], cmap='gray')
plt.title('%i' % train_data.train_labels[1])
plt.show()

# 数据加载机
train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# 把test_data转化成Variable, 拿出前2000个作为检测
test_data = dsets.MNIST(
    root='./mnist/',
    train=False,
    transform=transforms.ToTensor(),
)
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000] / 255
test_y = test_data.test_labels.numpy().squeeze()[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(  # 如果使用nn.RNN（），它几乎无法学习
            input_size=INPUT_SIZE,
            hidden_size=64,  # rnn隐藏单位
            num_layers=1,  # rnn层的数量
            batch_first=True,  # input＆output的批量大小是1维度。 例如 （batch，time_step，input_size）
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None表示零初始隐藏状态

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# 训练并测试神经网络
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):  # gives batch data
        b_x = Variable(x.view(-1, 28, 28))  # reshape x to (batch, time_step, input_size)
        b_y = Variable(y)  # batch y

        output = rnn(b_x)  # rnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:
            test_output = rnn(test_x)  # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)

# 打印前10个测试值
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')

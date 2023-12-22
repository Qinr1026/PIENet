#%% 建立模型，版本-1

# import library

import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from math import pi
import torch.nn.functional as F

#%% Laplace是一种单边衰减的复指数小波
def Laplace(p):  # p代表采样时间
    A = 0.08  # 归一化小波函数
    ep = 0.03   # 粘滞阻尼比
    tal = 0.1   # 时间参数
    f = 50  # 频率
    w = 2 * pi * f  # 固有频率
    q = torch.tensor(1 - pow(ep, 2))
    #  Laplace计算公式
    y = A * torch.exp((-ep / (torch.sqrt(q))) * (w * (p - tal))) * (-torch.sin(w * (p - tal)))
    return y

class Laplace_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1):

        super(Laplace_fast, self).__init__()

        if in_channels != 1:

            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels)).view(-1, 1)

        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels)).view(-1, 1)

    def forward(self, waveforms):

        time_disc = torch.linspace(0, 1, steps=int((self.kernel_size)))

        p1 = time_disc.cuda() - self.b_.cuda() / self.a_.cuda()

        laplace_filter = Laplace(p1)
        # print(laplace_filter)

        self.filters = (laplace_filter).view(self.out_channels, 1, self.kernel_size).cuda()

        return F.conv1d(waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1)

def channel_att(x, channel_num):
    _, c, _ = x.shape   # C为输入x的通道数   torch.Size([1, 256, 2687])
    abs_x = torch.abs(x)
    maxidx_x = torch.max(abs_x.sum(dim=2), dim=-1)[1]  # 最大值的索引
    max_x = x[:, maxidx_x]   # 输入的最大值  torch.Size([1, 1, 2687])
    max_x = max_x[:, 0, :]
    a = max_x.unsqueeze(1)

    sim = torch.cosine_similarity(x, max_x.unsqueeze(1), dim=2, eps=1e-08)

    simNorm = sim / sim.max()  # 权重置于0-1之间

    # 权重排序
    eff = math.floor(c * channel_num)  # 保留作为特征通道数
    value, index = torch.topk(simNorm, eff)  # 保留贡献度高的通道作为特征

    value = value.unsqueeze(dim=2)
    index_1 = index.unsqueeze(dim=2)  # 索引位置调整维度
    # print("index_1.shape", index_1.shape)  # [2,25]
    weight_x = torch.gather(x, 1, index_1)  # 取出这部分权重值
    # print("weight_x.shape", weight_x.shape)  # [2, 25, 1]

    eff_inputs = torch.take_along_dim(x, index_1, dim=1)  #[2, 25, 1026]

    out = eff_inputs * value

    return out, eff, simNorm
    ###### out:加权特征输出，eff：权重排序后的权重值，simNorm：原始各个通道的权重分布

#%% test module

class Model(nn.Module):
    def __init__(self, in_channel=1, out_channel=6):
        super(Model, self).__init__()
        self.cwt_features = nn.Sequential(
            Laplace_fast(128, 16)    # 输出128个通道 卷积核大小为16
        )

        self.input_size = 8087 # 2687 #8087
        self.hidden_size = 512
        self.num_layers = 3
        self.output_size = 6
        self.num_directions = 1  # 单向LSTM
        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.lin = torch.nn.Linear(self.hidden_size, self.output_size)

        self.dropout = nn.Dropout(0.5)

    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)  # d_k为query的维度
        # query:[batch, seq_len, hidden_dim], x.t:[batch, hidden_dim, seq_len]
        # 打分机制 scores: [batch, seq_len, seq_len]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        print("score: ", scores.shape)  # torch.Size([16, 469, 469])
        # 对最后一个维度 归一化得分
        alpha_n = F.softmax(scores, dim=-1)
        # 对权重化的x求和
        context = torch.matmul(alpha_n, x).sum(1)
        return context, alpha_n

    def forward(self, x):
        # print('input.shape', x.shape)  # input.shape torch.Size([16, 1, 4050])
        x1 = self.cwt_features(x)  # shape: [-1, 64, 1026]  可以看做是一个高64，长1026的时频图

        x2, eff, simNorm = channel_att(x1, 0.4)  # 通道注意力

        batch_size, seq_len = x2.shape[0], x2.shape[1]
        h_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(x2, (h_0.detach(), c_0.detach()))

        # 计算注意力权重
        query = self.dropout(output)
        attn_output, alpha_n = self.attention_net(output, query)

        pred = self.lin(attn_output)
        return x1, simNorm, alpha_n, pred

from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Model = Model()
Model = Model.to(device)

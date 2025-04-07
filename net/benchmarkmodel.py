import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils.tgcn import *
from net.utils.graph import Graph
from statsmodels.tsa.arima.model import ARIMA

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class LSTM3(nn.Module):
    def __init__(self, in_channels, graph_args, **kwargs):
        super(LSTM3, self).__init__()
        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.num_nodes = A.size(-1)
        self.in_channels = in_channels
        self.time_steps = 15
        self.hidden = 56 #hidden = res.size(1)*res.size(2)
        self.residual = nn.Sequential(nn.Conv2d(self.in_channels, 8,
                                                kernel_size=(3,1),stride=(2, 1)),
                                        nn.BatchNorm2d(8),)


        self.data_bn = nn.BatchNorm1d(self.in_channels * self.num_nodes)
        self.lstm = nn.LSTM(self.in_channels * self.time_steps, self.hidden, 3, batch_first=True)
        # self.inpre = nn.Sequential(nn.BatchNorm1d(self.in_channels * self.time_steps),nn.ReLU(inplace=True),)
        self.outpre = nn.Sequential(nn.Dropout(0.5, inplace=False),nn.ReLU(inplace=True),)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden, 1)

    def forward(self, x, A):
        B, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, V * C, T)
        x = self.data_bn(x)
        x = x.view(B, V, C, T)
        # x = x.permute(0, 3, 2, 1).contiguous() # (B,T,C,V)

        x_res = x.permute(0, 2, 3, 1).contiguous() #(B,C,T,V)
        res = self.relu(self.residual(x_res))
        res = res.permute(0, 3, 2, 1).contiguous()
        res = res.view(B, V, -1)
        # print(f"res shape = {res.shape}")
        x = x.permute(0, 3, 2, 1).contiguous() #(B, V, C, T)
        x = x.view(B, V, C * T)

        # x = self.inpre(x)
        x_out,_ = self.lstm(x)
        # print(f"x_out shape = {x_out.shape}") #(B,V, hidden)
        x_out = self.outpre(x_out)
        x_out = self.relu(x_out.contiguous().view(B, V, -1) + res)
        # x_out = x_out.contiguous().view(B, -1)
        x_out = self.fc(x_out)
        x_out = x_out.contiguous().view(B, V)
        x_clf = torch.zeros((B,2,V)).to(0)
        x_clf[:, 1, :] = 100.0
        return x_out, x_clf


class GRU(nn.Module):
    def __init__(self, in_channels, graph_args, **kwargs):
        super(GRU, self).__init__()
        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.num_nodes = A.size(-1)
        self.in_channels = in_channels
        self.time_steps = 15
        self.hidden = 12 #hidden = res.size(1)*res.size(2)
        self.drop = kwargs['dropout']
        self.residual = nn.Sequential(nn.Conv2d(self.in_channels, 2,
                                                kernel_size=(4,1),stride=(2, 1)),
                                        nn.BatchNorm2d(2),)


        self.data_bn = nn.BatchNorm1d(self.in_channels * self.num_nodes)
        self.gru = nn.GRU(self.in_channels * self.time_steps, self.hidden, 3, batch_first=True)
        # self.inpre = nn.Sequential(nn.BatchNorm1d(self.in_channels * self.time_steps),nn.ReLU(inplace=True),)
        self.outpre = nn.Sequential(nn.Dropout(self.drop, inplace=False),nn.ReLU(inplace=True),)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden, 1)

    def forward(self, x, A):

        # B, C, T, V = x.size()
        # x = x.permute(0, 3, 1, 2).contiguous()
        # x = x.view(B, V * C, T)
        # x = self.data_bn(x)
        # x = x.view(B, V, C, T)
        # x = x.permute(0, 3, 2, 1).contiguous()
        # x = x.view(B, T, C * V)

        B, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, V * C, T)
        x = self.relu(self.data_bn(x))
        x = x.view(B, V, C, T)
        # x = x.permute(0, 3, 2, 1).contiguous() # (B,T,C,V)

        x_res = x.permute(0, 2, 3, 1).contiguous() #(B,C,T,V)
        res = self.residual(x_res)
        res = res.permute(0, 3, 2, 1).contiguous()
        res = res.view(B, V, -1)
        # print(f"res shape = {res.shape}")
        x = x.permute(0, 3, 2, 1).contiguous() #(B, V, C, T)
        x = x.view(B, V, C * T)

        self.gru.flatten_parameters()
        x_out,_ = self.gru(x)
        # print(f"gru_out shape = {x_out.shape}")
        x_out = self.outpre(x_out)
        x_out = self.relu(x_out.contiguous().view(B, V, -1) + res)
        # print(f"before_fcx shape = {x_out.shape}")
        x_out = self.fc(x_out)
        # print(f"after_fcx shape = {x_out.shape}")
        x_out = x_out.contiguous().view(B, V)
        x_clf = torch.zeros((B,2,V)).to(0)
        x_clf[:, 1, :] = 100.0
        return x_out, x_clf

class Model_rt_gcn(nn.Module):

    def __init__(self, in_channels, graph_args,
                 **kwargs):
        super().__init__()
        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.out = kwargs['hidden'] #nas,sse 32

        # build networks
        # spatial_kernel_size = A.size(0) #3
        spatial_kernel_size = 1 #3

        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(2))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.drop = kwargs['dropout']
        # self.linear = nn.Linear(self.relation.size(2),1)
        self.rt_gcn_networks = nn.ModuleList((# st_gcn(in_channel,out_channel,kernel_size,stride,dropout,residual
            gcn_gru_4rtgcn(in_channels, self.out, kernel_size, A.size(-1), 2, self.drop, residual=True),))

        # initialize parameters for edge importance weighting
        self.edge_importance = [1] * len(self.rt_gcn_networks)
        # fcn for prediction
        self.fcn = nn.Conv2d(self.out, 1, kernel_size=1) #output_size=(N,1,1,V)
        self.fcn_clf = nn.Conv2d(self.out, 2, kernel_size=1)

    def forward(self, x, A):
        # data normalization
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)

        # forwad for uniform strategy
        for gcn, importance in zip(self.rt_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x,(x.size(2),1))
        x = x.view(N,-1,1,V)
        # prediction
        x_reg = self.fcn(x)
        x_reg = x_reg.view(x_reg.size(0), x_reg.size(3))

        # classification
        x_clf = self.fcn_clf(x)
        x_clf = x_clf.view(x_clf.size(0), x_clf.size(1), x_clf.size(3))

        return x_reg, x_clf

class gcn_gru_4rtgcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_nodes,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical_original(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.LeakyReLU(inplace=True,negative_slope=0.01)


    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A
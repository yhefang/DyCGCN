import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import *
from net.utils.graph import Graph
import torch_geometric.nn as pyg_nn
class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, drop):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_size,1), (stride,1))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), (stride,1))
        # self.res = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1, kernel_size)),nn.BatchNorm2d(out_channels),)

        self.drop = nn.Dropout(drop, inplace=True)
    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        # X = X.permute(0, 3, 1, 2)
        # temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        # out = F.relu(temp + self.conv3(X))

        X = X.permute(0, 2, 3, 1) #(B,out, T, V)
        # print(f"intcnX shape = {X.shape}")
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        temp = self.drop(temp)
        out = F.relu(temp)
        # Convert back from NCHW to NHWC
        # out = out.permute(0, 2, 3, 1).contiguous()
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=(1,1),stride=1)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        B, V, T, C = X.size()
        X = X.permute(0, 3, 2, 1).contiguous() #(B, V, T, C)->(B,C,T,V)
        # print(f"ingcnX shape = {X.shape}")
        X = self.conv(X)
        # print(f"aftgcnX shape = {X.shape}")
        # X = X.view(B * T, C, V)
        X = X.permute(0, 2, 1, 3).contiguous() #(B, T, C, V)
        # print(f"X shape = {X.shape}")
        # print(f"A shape = {A_hat.shape}")
        lfs = torch.matmul(X, A_hat) #(B,T,C,V)
        lfs = lfs.permute(0, 2, 1, 3).contiguous()

        # lfs = lfs.view(B, T, C, V)
        lfs = lfs.permute(0, 3, 1, 2).contiguous() #(B,V,T,C)

        return lfs

class AutoEncoder(nn.Module):
    def __init__(self, hidden2):
        super().__init__()

        # self.fc11 = nn.Conv2d(hidden2, int(hidden2/2), kernel_size=1)
        # self.fc12 = nn.Conv2d(int(hidden2/2), 1, kernel_size=1)
        # self.fc21 = nn.Conv2d(1, int(hidden2/2), kernel_size=1)
        # self.fc22 = nn.Conv2d(int(hidden2/2), hidden2, kernel_size=1)
        self.fc11 = nn.Conv2d(hidden2, int(hidden2/2), kernel_size=1)
        # self.fc12 = nn.Conv2d(int(hidden2/2), 1, kernel_size=1)
        # self.fc21 = nn.Conv2d(1, int(hidden2/2), kernel_size=1)
        self.fc22 = nn.Conv2d(int(hidden2/2), hidden2, kernel_size=1)

    def encode(self, x):
        e1 = F.relu(self.fc11(x))
        # e2 = F.relu(self.fc12(e1))
        return e1

    def decode(self, x):
        d1 = F.relu(self.fc22(x))
        # d2 = F.relu(self.fc22(d1))
        return d1

    def forward(self, x):
        # (B, V, T, -1)
        B, V, T, C = x.size()
        x = x.permute(0, 3, 2, 1).contiguous()
        # x = x.view(B,T,-1)
        hidden_repre = self.encode(x)
        autooutput = self.decode(hidden_repre)
        autooutput = autooutput.permute(0, 3, 2, 1).contiguous()
        autooutput = autooutput.view(B, V, T, C)
        # autooutput = autooutput.permute(0,2,3,1).contiguous()
        return autooutput

class STGCN_withoutdynet(nn.Module):
    """
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, in_channels, graph_args, **kwargs):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN_withoutdynet, self).__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        A = A.view(A.size(1),A.size(2))
        self.register_buffer('A', A)

        self.num_nodes = A.size(-1)
        self.in_channels = in_channels -1
        self.hidden = kwargs['hidden'] #8
        self.hidden2 = kwargs['hidden2']
        self.out_channels = kwargs['out_channels']
        self.kernel = kwargs['kernel']
        self.stride = kwargs['stride']
        self.drop = kwargs['dropout']
        self.autoencoder = AutoEncoder(self.hidden2)
        self.block1 = STGCNBlock(in_channels=self.in_channels, out_channels=self.hidden,
                                 spatial_channels=16, num_nodes=self.num_nodes)
        # self.block2 = STGCNBlock(in_channels=64, out_channels=64,
        #                          spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=self.hidden + self.hidden2, out_channels=self.out_channels,
                                       kernel_size=self.kernel, stride=self.stride, drop=self.drop)

        self.res = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels,
                                           kernel_size=(self.kernel,1), stride=(self.stride,1)),
                                 nn.BatchNorm2d(self.out_channels), )
        self.relu = nn.ReLU()
        # self.fully = nn.Linear(self.out_channels * (15 - self.kernel + 1), 1)
        self.gru = nn.GRU(self.hidden, self.hidden2, 1, batch_first=True)
        self.fcn = nn.Conv2d(self.out_channels, 1, kernel_size=1)
        self.fcn_clf = nn.Conv2d(self.out_channels, 2, kernel_size=1)
        self.batch_norm = nn.BatchNorm1d(self.in_channels * A.size(-1))
    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        X = X[:, 0:3, :, :]
        X = X.permute(0, 3, 2, 1).contiguous()
        B, V, T, C = X.size()

        x_res = X.permute(0, 3, 2, 1).contiguous() #(B,C,T,V)
        x_res = x_res.view(B, C, T, V)

        res = self.res(x_res) #(B,out, T-kernel, V)

        out3 = self.block1(X, self.A) #(B,V,T,C)

        out3 = out3.permute(0, 1, 3, 2).contiguous()  # (B,V,T,hid)
        out3 = out3.view(B*V, T, -1)
        self.gru.flatten_parameters()
        h_out, _ = self.gru(out3)  # (BV,T,hid2)
        # 恢复时间维度
        out3 = out3.view(B, V, T, -1)
        h_out = h_out.view(B, V, T, -1)
        h_out = self.autoencoder(h_out)
        out3 = torch.cat((out3, h_out), dim=3)
        out3 = out3.permute(0, 1, 3, 2).contiguous()

        out3 = self.last_temporal(out3)

        out3 = self.relu(out3 + res)

        # global pooling
        out3 = F.avg_pool2d(out3,(out3.size(2),1))
        out = out3.view(B,-1,1,V)
        # out = out.permute(0,2,1).contiguous()
        out1 = self.fcn(out)
        out1 = out1.view(B,V)

        # classification
        x_clf = self.fcn_clf(out)
        x_clf = x_clf.view(x_clf.size(0), x_clf.size(1), x_clf.size(3))

        return out1, x_clf
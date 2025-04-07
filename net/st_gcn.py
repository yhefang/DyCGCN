import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils.graph import Graph

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
        temp = self.conv1(X) +torch.sigmoid(self.conv2(X))
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
        # self.temporal1 = TimeBlock(in_channels=in_channels,
        #                            out_channels=out_channels)
        # self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
        #                                              spatial_channels))
        # self.temporal2 = TimeBlock(in_channels=spatial_channels,
        #                            out_channels=out_channels)
        # self.res = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1, 3)), nn.BatchNorm2d(out_channels), )
        # self.batch_norm = nn.BatchNorm2d(num_nodes)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        B, V, T, C = X.size()
        X = X.permute(0, 3, 2, 1).contiguous()
        # print(f"ingcnX shape = {X.shape}")
        X = self.conv(X)
        # X = X.view(B * T, C, V)
        # print(f"aftconvgcnX shape = {X.shape}")
        # print(f"A shape = {A_hat.shape}")

        # t = self.temporal1(X)
        # lfs = torch.einsum("ij,jklm->kilm", [A_hat, X.permute(1, 0, 2, 3)])
        lfs = torch.matmul(X, A_hat) #(B,T,C,V)
        # lfs = lfs.view(B, T, C, V)
        lfs = lfs.permute(0, 3, 1, 2).contiguous() #(B,V,T,C)
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        # t2 = F.relu(torch.matmul(lfs, self.Theta1))
        # t3 = self.temporal2(t2)
        # return self.batch_norm(lfs)
        return lfs


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
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
        super(STGCN, self).__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        A = A.view(A.size(1),A.size(2))
        self.register_buffer('A', A)

        self.num_nodes = A.size(-1)
        self.in_channels = in_channels -1
        self.hidden = kwargs['hidden']
        self.out_channels = kwargs['out_channels']
        self.kernel = kwargs['kernel']
        self.stride = kwargs['stride']
        self.drop = kwargs['dropout']

        self.block1 = STGCNBlock(in_channels=self.in_channels, out_channels=self.hidden,
                                 spatial_channels=16, num_nodes=self.num_nodes)
        # self.block2 = STGCNBlock(in_channels=64, out_channels=64,
        #                          spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=self.hidden, out_channels=self.out_channels,
                                       kernel_size=self.kernel, stride=self.stride, drop=self.drop)

        self.res = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels,
                                           kernel_size=(self.kernel,1), stride=(self.stride,1)),
                                 nn.BatchNorm2d(self.out_channels), )
        self.relu = nn.ReLU()
        # self.fully = nn.Linear(self.out_channels * (15 - self.kernel + 1), 1)
        self.fcn = nn.Conv2d(self.out_channels, 1, kernel_size=1)

        self.batch_norm = nn.BatchNorm1d(self.in_channels * A.size(-1))
    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        X = X[:, 0:3, :, :]
        X = X.permute(0, 3, 2, 1).contiguous()
        # print(f"X shape = {X.shape}")
        B, V, T, C = X.size()
        # X = X.permute(0, 3, 1, 2).contiguous()
        # X = X.view(B, V * C, T)
        # X = self.batch_norm(X)
        # X = X.view(B, V, T, C)

        x_res = X.permute(0, 3, 2, 1).contiguous() #(B,C,T,V)
        x_res = x_res.view(B, C, T, V)
        # print(f"beforeres shape = {x_res.shape}")
        res = self.res(x_res) #(B,out, T-kernel, V)
        # print(f"afterres shape = {res.shape}")
        out3 = self.block1(X, self.A) #(B,V, out,T)
        # print(f"gcn_out shape = {out3.shape}")
        # out2 = self.block2(out1, self.A)
        out3 = self.last_temporal(out3)
        # print(f"tcn_out shape = {out3.shape}")
        # print(f"gru_out shape = {x_out.shape}")

        out3 = self.relu(out3 + res)

        # out3 = out3.permute(0, 3, 2, 1).contiguous()

        # global pooling
        out3 = F.avg_pool2d(out3,(out3.size(2),1))
        # out3 = out3.view(B,-1,V)
        # print(f"avgpool_out shape = {out3.shape}")

        out = out3.view(B,-1,1,V)
        # out = out.permute(0,2,1).contiguous()

        out = self.fcn(out)
        out = out.view(B,V)
        # out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        #
        # out4 = out4.view(out4.size(0), out4.size(1))
        # clf = self.clf(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        # clf = clf.view(clf.size(0), clf.size(2), clf.size(1))
        # out3 = F.relu(out3, inplace=True)
        # out3 = out3.permute(0, 3, 2, 1).contiguous()

        # prediction
        # out = self.fully(out3.reshape(out3.shape[0],-1))

        x_clf = torch.zeros((B,2,V)).to(0)
        x_clf[:, 1, :] = 100.0

        return out, x_clf

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils.graph import Graph

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
        temp = self.conv1(X) +torch.sigmoid(self.conv2(X))
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
        # self.temporal1 = TimeBlock(in_channels=in_channels,
        #                            out_channels=out_channels)
        # self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
        #                                              spatial_channels))
        # self.temporal2 = TimeBlock(in_channels=spatial_channels,
        #                            out_channels=out_channels)
        # self.res = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1, 3)), nn.BatchNorm2d(out_channels), )
        # self.batch_norm = nn.BatchNorm2d(num_nodes)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        B, V, T, C = X.size()
        X = X.permute(0, 3, 2, 1).contiguous()
        # print(f"ingcnX shape = {X.shape}")
        X = self.conv(X)
        # X = X.view(B * T, C, V)
        # print(f"aftconvgcnX shape = {X.shape}")
        # print(f"A shape = {A_hat.shape}")

        # t = self.temporal1(X)
        # lfs = torch.einsum("ij,jklm->kilm", [A_hat, X.permute(1, 0, 2, 3)])
        lfs = torch.matmul(X, A_hat) #(B,T,C,V)
        # lfs = lfs.view(B, T, C, V)
        lfs = lfs.permute(0, 3, 1, 2).contiguous() #(B,V,T,C)
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        # t2 = F.relu(torch.matmul(lfs, self.Theta1))
        # t3 = self.temporal2(t2)
        # return self.batch_norm(lfs)
        return lfs


class STGCN2(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
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
        super(STGCN2, self).__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        A = A.view(A.size(1),A.size(2))
        self.register_buffer('A', A)

        self.num_nodes = A.size(-1)
        self.in_channels = in_channels -1
        self.hidden = kwargs['hidden']
        self.out_channels = kwargs['out_channels']
        self.kernel = kwargs['kernel']
        self.stride = kwargs['stride']
        self.drop = kwargs['dropout']

        self.block1 = STGCNBlock(in_channels=self.in_channels, out_channels=self.hidden,
                                 spatial_channels=16, num_nodes=self.num_nodes)
        # self.block2 = STGCNBlock(in_channels=64, out_channels=64,
        #                          spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=self.hidden, out_channels=self.out_channels,
                                       kernel_size=self.kernel, stride=self.stride, drop=self.drop)

        self.res = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels,
                                           kernel_size=(self.kernel,1), stride=(self.stride,1)),
                                 nn.BatchNorm2d(self.out_channels), )
        self.relu = nn.ReLU()
        # self.fully = nn.Linear(self.out_channels * (15 - self.kernel + 1), 1)
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
        # print(f"X shape = {X.shape}")
        B, V, T, C = X.size()
        # X = X.permute(0, 3, 1, 2).contiguous()
        # X = X.view(B, V * C, T)
        # X = self.batch_norm(X)
        # X = X.view(B, V, T, C)

        x_res = X.permute(0, 3, 2, 1).contiguous() #(B,C,T,V)
        x_res = x_res.view(B, C, T, V)
        # print(f"beforeres shape = {x_res.shape}")
        res = self.res(x_res) #(B,out, T-kernel, V)
        # print(f"afterres shape = {res.shape}")
        out3 = self.block1(X, self.A) #(B,V, out,T)
        # print(f"gcn_out shape = {out3.shape}")
        # out2 = self.block2(out1, self.A)
        out3 = self.last_temporal(out3)
        # print(f"tcn_out shape = {out3.shape}")
        # print(f"gru_out shape = {x_out.shape}")

        out3 = self.relu(out3 + res)

        # out3 = out3.permute(0, 3, 2, 1).contiguous()

        # global pooling
        out3 = F.avg_pool2d(out3,(out3.size(2),1))
        # out3 = out3.view(B,-1,V)
        # print(f"avgpool_out shape = {out3.shape}")

        out = out3.view(B,-1,1,V)
        # out = out.permute(0,2,1).contiguous()

        out1 = self.fcn(out)
        out1 = out1.view(B,V)
        # out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        #
        # out4 = out4.view(out4.size(0), out4.size(1))
        # clf = self.clf(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        # clf = clf.view(clf.size(0), clf.size(2), clf.size(1))
        # out3 = F.relu(out3, inplace=True)
        # out3 = out3.permute(0, 3, 2, 1).contiguous()

        # prediction
        # out = self.fully(out3.reshape(out3.shape[0],-1))

        # classification
        x_clf = self.fcn_clf(out)
        x_clf = x_clf.view(x_clf.size(0), x_clf.size(1), x_clf.size(3))

        return out1, x_clf




class TimeBlock(nn.Module):
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
        # self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), (stride,1))
        self.drop = nn.Dropout(drop, inplace=False)
    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels) (B,V,T,C)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        X = X.permute(0, 3, 2, 1) #(B,hid,T,V)
        temp = self.conv1(X)
        temp = self.drop(temp)
        out = F.relu(temp)

        return out

class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, out_channels,):
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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=1)


    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        B, V, T, C = X.size()
        X = X.permute(0, 3, 2, 1).contiguous() #(B,hid,T,V)
        X = self.conv(X) #(B,hid,T,V) A(B,T,V,V)
        X = X.permute(0, 2, 1, 3).contiguous()
        x_out = torch.matmul(X, A_hat) #(B,T,hid,V)
        x_out = x_out.view(B,T,-1,V)
        # lfs = torch.matmul(X, A_hat) #(B,T,hid,V)
        x_out = x_out.permute(0, 3, 1, 2).contiguous() #(B,V,T,hid)
        return x_out

class DynGCNGRU(nn.Module):
    def __init__(self, in_channels, graph_args, **kwargs):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(DynGCNGRU, self).__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        A = A.view(A.size(1),A.size(2))
        self.register_buffer('A', A)

        self.num_nodes = A.size(-1)
        self.in_channels = in_channels
        self.hidden1 = 4
        self.hidden2 = 3
        self.out_channels = 5
        self.kernel = 9
        self.stride = 1
        self.drop = kwargs['dropout']

        self.block1 = STGCNBlock(in_channels=self.in_channels, out_channels=self.hidden1,)
        self.temporal = TimeBlock(in_channels=self.hidden1 + self.hidden2, out_channels=self.out_channels,
                                       kernel_size=self.kernel, stride=self.stride, drop=self.drop)

        self.res = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels,
                                           kernel_size=(self.kernel,1), stride=(self.stride,1)),
                                 nn.BatchNorm2d(self.out_channels), )
        self.relu = nn.ReLU()
        # self.fully = nn.Linear(self.out_channels * (15 - self.kernel + 1), 1)
        self.fcn = nn.Conv2d(self.out_channels, 1, kernel_size=1)
        self.fcn_clf = nn.Conv2d(self.out_channels, 2, kernel_size=1)
        self.gru = nn.GRU(self.hidden1, self.hidden2, 1, batch_first=True)

        self.batch_norm = nn.BatchNorm1d(self.in_channels * self.num_nodes)
    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        # X = X[:, 0:3, :, :]
        # B, C, T, V = x.size()
        X = X.permute(0, 3, 2, 1).contiguous() #(B,V,T,C)
        B, V, T, C = X.size()
        X = X.permute(0, 3, 1, 2).contiguous()
        X = X.view(B, V * C, T)
        X = self.batch_norm(X)
        X = X.view(B, V, T, C)


        x_res = X.permute(0, 3, 2, 1).contiguous() #(B,C,T,V)
        x_res = x_res.view(B, C, T, V)
        res = self.res(x_res) #(B,out,T-kernel,V)
        out3 = self.block1(X, A_hat) #(B,V,T,hid)

        # B, V, T, C = out3.size()
        out3 = out3.view(B * V, T, -1) #(BV,T,hid)
        self.gru.flatten_parameters()
        h_out, _ = self.gru(out3) #(BV,T,hid2)
        # 恢复时间维度
        out3 = out3.view(B, V, T, -1)
        h_out = h_out.view(B, V, T, -1)
        # print(f"aftgru_x shape = {h_out.shape}") #(B, out, T, V)
        # x = x.view(B, V, T, -1)
        # h_out = h_out.permute(0, 3, 2, 1).contiguous() #（B，hid2，T，V）
        out3 = torch.cat((out3, h_out), dim=3)
        out3 = self.temporal(out3) #(B,out,T-kernel,V)

        out3 = self.relu(out3 + res)
        #要把out3变换一下
        # ou3 = out3.permute(0, 3, 1, 2).contiguous()
        # global pooling
        out3 = F.avg_pool2d(out3,(out3.size(2),1))

        out = out3.view(B,-1,1,V)

        out1 = self.fcn(out)
        out1 = out1.view(B,V)

        # classification
        x_clf = self.fcn_clf(out)
        x_clf = x_clf.view(x_clf.size(0), x_clf.size(1), x_clf.size(3))

        return out1, x_clf
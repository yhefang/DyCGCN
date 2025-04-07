import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils.graph import Graph

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, num_nodes, time_steps):
        super().__init__()
        self.fc11 = nn.Linear(in_channels * num_nodes, num_nodes * 2)
        self.fc12 = nn.Linear(num_nodes * 2, num_nodes)
        self.fc21 = nn.Linear(num_nodes, 2 * num_nodes)
        self.fc22 = nn.Linear(2 * num_nodes, in_channels * num_nodes)

    def encode(self, x):
        e1 = F.relu(self.fc11(x))
        e2 = F.relu(self.fc12(e1))
        return e2

    def decode(self, x):
        d1 = F.relu(self.fc21(x))
        d2 = F.relu(self.fc22(d1))
        return d2

    def forward(self, x):
        B, T, C, V = x.size()
        # x = x.permute(0,3,1,2).contiguous()
        x = x.view(B,T,-1)
        hidden_repre = self.encode(x)
        autooutput = self.decode(hidden_repre)
        autooutput = autooutput.view(B, T, C, V)
        # autooutput = autooutput.permute(0,2,3,1).contiguous()
        return autooutput, hidden_repre

class STBlock(nn.Module):
    def __init__(self, in_channels, num_nodes, time_steps):
        super(STBlock, self).__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.st_gcn = GCN(self.in_channels, self.num_nodes)
        self.st_lstm = MultiVariateLSTM(num_nodes)

    def forward(self, input_feature, adj):
        B, T, C2, V = input_feature.size()
        # input_feature = input_feature.view(B, T * self.in_channels, self.num_nodes)
        lstm_inputs = []
        for i in range(T):
            # print(f"adj.shape={adj.shape}")
            # input_feature = input_feature.permute(0, 3, 2, 1).contiguous()
            # print(f"input feature.shape ={input_feature.shape}")
            out1 = self.st_gcn(adj, input_feature[:, i, :, :]) #(B,C2,V)
            # print(f"temp out shape={out1.shape}")
            if i == 0:
                lstm_inputs = out1
            else:
                lstm_inputs = torch.vstack((lstm_inputs, out1))

        lstm_inputs = lstm_inputs.view(T, B, -1) #suppose to be (B,T,V)
        lstm_inputs = lstm_inputs.transpose(0, 1)
        lstm_inputs = lstm_inputs.view(B, T, V)
        # print(f"lstm_input_shape={lstm_inputs.shape}")
        lstm_outputs = self.st_lstm(lstm_inputs)
        return lstm_outputs
        # return lstm_inputs

class MultiVariateLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size  # node number, number of variable
        self.hidden_size = int(input_size/2)  # number of node in hidden layer, set at will
        self.num_layers = 1  # how many lstm layer are stacked together
        self.output_size = input_size  #
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        B, T, V = input_seq.size()
        h_0 = torch.randn(self.num_layers, B, self.hidden_size).to(input_seq.device)
        c_0 = torch.randn(self.num_layers, B, self.hidden_size).to(input_seq.device)
        # print(input_seq.size())
        seq_len = input_seq.shape[1]  # here is cf.seq_len   15
        # input (batch_size, seq_len, input_size)  (32, 30, 207)
        input_seq = input_seq.view(B, seq_len, self.input_size)  # do not need sometimes if the input_seq shape is correct
        # output  (batch_size, seq_len, num_directions * hidden_size)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(input_seq, (h_0, c_0))  # here _ represents (h_n, c_n)
        # print("output.size=", output.size())
        output = output.contiguous().view(B * seq_len, self.hidden_size)  # (32*30, 200)
        # print("output.size 2=", output.size())
        pred = self.fc(output)
        # print("pred shape = ", pred.shape)  #(960,207)
        pred = pred.view(B, seq_len, -1)
        # print("pred shape 2 = ", pred.shape)  #(32,6,207)
        return pred

class GraphConvolution(nn.Module):
    def __init__(self, in_channels, num_nodes):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.linear = nn.Linear(self.in_channels * self.num_nodes * 2, num_nodes)

    def forward(self, adj, features):
        B,C2,V = features.size()
        features = features.permute(2, 1, 0).contiguous()
        # adj_reshaped = adj.unsqueeze(-1)
        # print(f"adj shape={adj.shape}, features shape={features.shape}")
        out = torch.einsum('ij,jkl->ikl', adj, features)  # graph = f(A_hat*W)
        # print(f'out.shape={out.shape}') #(V,2C,B)
        # out = torch.transpose(out,1,0)
        out = out.permute(2, 1, 0).contiguous()
        out = out.view(B,-1)
        out = self.linear(out)
        # print("gcn out shape = ", out.shape)
        return out


class GCN(torch.nn.Module):
    def __init__(self,in_channels, num_nodes):
        super(GCN, self).__init__()
        self.in_channels = in_channels
        self.gcn1 = GraphConvolution(self.in_channels, num_nodes)
        # self.gcn2 = GraphConvolution(num_nodes, output_size)

        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.01)
    def forward(self, adj, features):
        # features (B,C2,V)

        # print(f"adj shape {adj.shape}, features shape {features.shape}")
        out = self.gcn1(adj, features)
        out = self.relu(out)
        # print(f"out shape = {out.shape}")
        # out = self.gcn2(adj, out.transpose(1, 0))
        return out

class SASTGCN(nn.Module):
    def __init__(self, in_channels, graph_args, **kwargs):
        super(SASTGCN, self).__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        A = A.view(A.size(1), A.size(2))
        self.register_buffer('A', A)
        num_nodes = A.size(-1)
        self.in_channels = in_channels -1
        self.num_nodes = num_nodes
        self.time_steps = 15
        self.hidden = kwargs['hidden']
        self.out_channels = kwargs['out_channels']
        self.kernel = kwargs['kernel']
        self.stride = kwargs['stride']
        self.drop = kwargs['dropout']
        self.res = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels ,
                                                kernel_size=(self.kernel,1), stride=(self.stride,1)),
                                        nn.BatchNorm2d(self.out_channels),)

        self.stblock = STBlock(self.in_channels, self.num_nodes, self.time_steps)
        self.selfsupervision = AutoEncoder(self.in_channels, self.num_nodes, self.time_steps)
        # self.fc = nn.Linear(in_features=self.time_steps*self.num_nodes, out_features=num_nodes)
        # self.adj = adj
        self.relu = nn.ReLU()
        self.fcn = nn.Conv2d(self.time_steps, 1, kernel_size=1)
        self.fcn_clf = nn.Conv2d(self.time_steps, 2, kernel_size=1)
        self.data_bn = nn.BatchNorm1d(self.in_channels * num_nodes)
    def forward(self, x, adj):
        x = x[:, 0:3, :, :]
        B, C, T, V = x.size()

        # x = x.permute(0, 3, 1, 2).contiguous()
        # x = x.view(B, V * C, T)
        # x = self.data_bn(x)
        # x = x.view(B, V, C, T)
        # x = x.permute(0, 3, 2, 1).contiguous()
        x_res = x #(B,C,T,V)
        res = self.res(x_res) #(B,out, T-kernel, V)

        # print(f"afterres shape = {res.shape}")

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T, C, V)
        autooutput, hidden_fea = self.selfsupervision(x) #autooutput (B,T,CV),hidden shape (B,T,V)
        # print(f"autooutput shape = {autooutput.shape}") #(B, T, C, V)
        # print(f"hidden_fea shape = {hidden_fea.shape}")

        bias = x - autooutput #bias (B,T,CV)

        x = x.view(B,T,C,V)
        bias = bias.view(B,T,C,V)

        new_x = torch.cat((x, bias), 2) # (B,T,2C,V)
        # print(f"new_x.shape={new_x.shape}")


        st_out = self.stblock(new_x, self.A)
        # print(f"st_out.shape={st_out.shape}") #(B,T,V)
        # print(f"hidden_fea.shape={hidden_fea.shape}")

        st_out = self.relu(st_out + res.view(B,-1,V))
        # print(f"st_out.shape={st_out.shape}") #(B,T,V)
        st_out = st_out.view(B,-1,1,V)
        # st_out = st_out.view(B, -1)
        # hidden_fea = hidden_fea.view(B, -1)
        # new_feature = torch.cat((st_out, hidden_fea), 1)
        # print(f"new_feature.shape={new_feature.shape}")
        out = self.fcn(st_out)
        out = out.view(B, V)
        # print(f"out.shape={out.shape}")

        # classification
        x_clf = torch.zeros((B,2,V)).to(0)
        x_clf[:, 1, :] = 100.0

        # # classification
        # x_clf = self.fcn_clf(st_out)
        # x_clf = x_clf.view(x_clf.size(0), x_clf.size(1), x_clf.size(3))
        return out, x_clf

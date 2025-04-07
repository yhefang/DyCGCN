# The based unit of graph convolutional networks.

import torch
import torch.nn as nn


class Loss(nn.Module):

    r"""The loss function including mean square error and ranking loss.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
 
    """

    def __init__(self):
        super().__init__()
        self.loss1 = nn.MSELoss()
        self.relu = nn.ReLU()

    def forward(self, prediction, groud_truth, alpha=0.5):
        N, V = groud_truth.size()
        prediction = prediction.view(N, V, 1)
        groud_truth = groud_truth.view(N, V, 1)
        one_all = torch.ones((V,1), dtype=torch.float32, device='cuda')
        # one_all = torch.tensor(one_all, dtype=torch.float32, requires_grad=False, device='cuda')
        pred_pw_dif = torch.sub(
            torch.matmul(prediction, one_all.permute(1,0)),
            torch.matmul(one_all, prediction.permute(0,2,1))
            )
        gt_pw_dif = torch.sub(
            torch.matmul(one_all, groud_truth.permute(0,2,1)),
            torch.matmul(groud_truth, one_all.permute(1,0))
            )
        self.loss2 = torch.mean(
            self.relu(
                    torch.mul(pred_pw_dif, gt_pw_dif)
                )
            )
        self.loss = self.loss1(prediction, groud_truth) + alpha * self.loss2
        return self.loss


class Loss_new(nn.Module):
    r"""The loss function including mean square error and ranking loss.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel

    """

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, prediction_value, groud_truth, prediction_prob, alpha=0.5):
        N, V = groud_truth.size()
        prediction_value = prediction_value.view(N, V, 1)
        groud_truth = groud_truth.view(N, V, 1)
        prediction_prba = prediction_prob.view(N, V, 1)
        # self.loss1 = 1/(N * V) * torch.norm(prediction_value-groud_truth)
        self.loss1 = nn.MSELoss()

        # self.loss2 = torch.sum(
        #     self.relu(
        #         -(prediction_value-groud_truth)
        #     ) * prediction_prba
        # )
        # self.loss2 = torch.mean(
        #     self.relu(
        #         -(prediction_value-groud_truth)
        #     ) * prediction_prba
        # )
        # 低估修正
        self.loss2 = torch.mean(self.relu(-(prediction_value-groud_truth)))
        # 高估修正
        # self.loss2 = torch.mean(self.relu(prediction_value-groud_truth))
        # 对称修正
        # self.loss2 = torch.mean(abs(prediction_value-groud_truth))




        # self.loss = (1 - alpha) * self.loss1 + alpha * self.loss2
        self.loss = (1 - alpha) * self.loss1(prediction_value, groud_truth) + alpha * self.loss2

        return self.loss

class Loss_clf(nn.Module):
    r"""The loss function for classify.
    Args:

    """
    def __init__(self):
        super().__init__()
        self.cel = nn.CrossEntropyLoss()

    def forward(self, clf_result, groud_truth):
        # 大于0时表示收益率为正，上涨
        # clf_result = torch.where(clf_result > 0.5, torch.tensor(1).to(clf_result.device), torch.tensor(0).to(clf_result.device))
        groud_truth = torch.where(groud_truth > 0, torch.tensor(1).to(groud_truth.device), torch.tensor(0).to(groud_truth.device))
        # print(groud_truth)
        groud_truth = groud_truth.long()
        self.loss = self.cel(clf_result, groud_truth)

        return self.loss
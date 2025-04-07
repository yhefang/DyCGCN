#!/usr/bin/env python
# pylint: disable=W0201
import sys
import time
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
# torch
import torch
import torch.nn as nn
import torch.optim as optim

# loss
from net.loss import Loss
from net.loss import Loss_new, Loss_clf
# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class
from sklearn.metrics import r2_score
from processor.processor import Processor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        '''
        self.model_clf = self.io.load_model(self.arg.model_clf,
                                        **(self.arg.model_args))
        
        '''
        # self.loss = nn.CrossEntropyLoss()
        # self.loss = nn.MSELoss()
        # self.loss = Loss()

        self.loss = Loss_new()
        self.loss_clf = Loss_clf()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            # self.lr = self.arg.base_lr
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr

    def show_topk(self, k):
        rank = self.result.argsort() # self.result shape (batch_size, num_node)
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.6f}%'.format(k, 100 * accuracy))


    def show_return_ration(self, k):
        # 这里需要修改为计算概率的EIR
        # 这里应该对根据预测结果计算得到的收益率计算期望后再排名
        # calculate the return-ration of the top k stocks
        self.bt_k = 1.0

        # # 单纯收益的情况
        # pre_ir = (self.result[1:,:] - self.result[0:-1,:])/self.result[0:-1,:]
        # print(pre_ir)
        # 根据收益率正负情况乘单独概率
        # print(self.prob[:, :])
        pre_ir = (self.result[1:,:] - self.result[0:-1,:])/self.result[0:-1,:]
        pre_ir = np.where(pre_ir > 0, pre_ir * self.prob[1:, :], pre_ir * (1 - self.prob[1:, :]))
        # print(pre_ir.shape)
        # print(self.label.shape)
        #
        # # 乘概率之差的情况 这里的prob代表上涨概率
        # pre_ir = ((self.result[1:,:] - self.result[0:-1,:])/self.result[0:-1,:])*abs((0.5-self.prob[1:,:]))*2
        # print(pre_ir)
        for i in range(pre_ir.shape[0]):
            rank_pre = np.argsort(self.result[i])

            pre_topk = set()
            for j in range(1, pre_ir.shape[1] + 1):
                cur_rank = rank_pre[-1 * j]
                if len(pre_topk) < k:
                    pre_topk.add(cur_rank)

            # back testing on top k
            return_ration_topk = 0
            for index in pre_topk:

                # 这里用于计算期望收益率用的是收益率排名最高的股票的真实收益率
                # return_ration_topk += self.label[i][index]*(0.5-self.prob[i][index])*2
                return_ration_topk += self.label[i+1][index]
                # return_ration_topk += self.label[i][index]
            return_ration_topk /= k
            with open('EIR_{}.txt'.format(k),'a+') as f:
                f.write(str(round((return_ration_topk),2)))
                f.write('\n')
                f.close()
            self.bt_k += return_ration_topk
            # print(i,return_ration_topk,self.bt_k)
        # self.io.print_log('\tTop{} return ratio: {:.2f}'.format(k, bt_k))

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        start_time = time.time()
        for data, closing_price, label, adj in loader:

            # get data
            # print(label.shape,label[:5,:5])
            # exit()
            data = data.float().to(self.dev)
            closing_price = closing_price.float().to(self.dev)
            label = label.float().to(self.dev)
            adj = adj.float().to(self.dev)
            # forward
            # output = self.model(data)
            output, clf_result = self.model(data, adj)
            # prediction = torch.div(torch.sub(output, closing_price), closing_price)
            # prediction_prob = torch.sigmoid(clf_result)
            # print(output)
            # print(clf_result)
            prediction_prob = torch.softmax(clf_result, dim=1)[:, 0:1, :].view((clf_result.size(0), clf_result.size(2)))

            # loss_tol = self.loss(output, closing_price)

            loss = self.loss(output, closing_price, prediction_prob, 0.5)
            loss_clf = self.loss_clf(clf_result, label)
            loss_tol = loss + loss_clf


            # backward-reg
            self.optimizer.zero_grad()
            loss_tol.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss_tol.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_log('Time consumption:{:.4f}s'.format(time.time()-start_time))
        with open('time_consume.txt','a+') as f:
            f.write(str(round((time.time()-start_time),2)))
            f.write('\n')
            f.close()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        cp_frag = []
        prob_frag = []
        for data, closing_price, label, adj in loader:
            
            # get data
            data = data[:,:,:,:].float().to(self.dev)
            closing_price = closing_price.float().to(self.dev)
            label = label.float().to(self.dev)
            adj = adj.float().to(self.dev)

            # print(closing_price[:5,:10])
            # inference
            with torch.no_grad():
                # output输出预测价格，还需要加一个输出预测概率的模型
                # output = self.model(data)
                output, clf_result = self.model(data, adj)
                # print(output)
                # print(clf_result)
                # print(output[:5,:10])
                # predication得到的是收益率
                # prediction = torch.div(torch.sub(output, closing_price), closing_price)
                # prediction_prob = torch.sigmoid(clf_result)
                prediction_prob = torch.softmax(clf_result, dim=1)[:, 0:1, :].view((clf_result.size(0), clf_result.size(2)))

            result_frag.append(output.data.cpu().numpy())
            prob_frag.append(prediction_prob.data.cpu().numpy())
            # get loss
            if evaluation:
                # loss = self.loss(output, closing_price)
                loss = self.loss(output, closing_price, prediction_prob, 0.5)

                loss_value.append(loss.item())
                cp_frag.append(closing_price.data.cpu().numpy())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)



        self.prob = np.concatenate(prob_frag)
        # print(self.prob)
        # print(self.prob.shape)
        self.cp = np.concatenate(cp_frag)
        np.savez("forgraph/sp500-g.npz", result=self.result, cp=self.cp)
        '''
        row_means1 = np.mean(self.result, axis=1)
        row_means2 = np.mean(self.cp, axis=1)
        # 假设 row_means1 和 row_means2 是两个长度相同的一维数组
        x = range(len(row_means1))
        np.savez("forgraph/case-our.npz", result=self.result, cp=self.cp)
        # 创建一个新的图形和轴
        fig, ax = plt.subplots(figsize=(12, 4))

        # 绘制折线图
        ax.plot(x, row_means2,'--',label='True closing price', alpha=0.7, linewidth=0.7,marker='o',markersize=0.1)
        ax.plot(x, row_means1, label='Prediction result', alpha=0.7, linewidth=0.8,marker='o',markersize=0.1)

        # 添加图例
        ax.legend()

        # 设置横轴标签
        ax.set_xlabel('Time')

        # 设置纵轴标签
        ax.set_ylabel('Closing price')

        # 设置图表标题
        # ax.set_title('Line Plot of Row Means')
        # 显示图形
        plt.show()
        '''
        self.label = np.concatenate(label_frag)
        if evaluation:
            rmse = 0.0
            for i in range(self.result.shape[0]):
                rmse += np.sum(np.square(self.result[i] - self.cp[i])) / len(self.cp[i])
            r2 = r2_score(np.mean(self.cp, axis=1), np.mean(self.result, axis=1))
            self.epoch_info['RMSE'] = np.sqrt(rmse / self.result.shape[0])
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.epoch_info['R2'] = r2
            self.show_epoch_info()

            if self.epoch_info['mean_loss'] < self.best_performance['mean_loss']:
                # save the model
                self.io.save_model(self.model, 'best_model.pt')
                self.best_performance['mean_loss'] = self.epoch_info['mean_loss']
                self.best_performance['RMSE'] = self.epoch_info['RMSE']
                self.best_performance['R2'] = self.epoch_info['R2']
                # show top-k return ration
                for k in self.arg.show_topk:
                    self.show_return_ration(k)
                    self.best_performance['top'+str(k)] = self.bt_k
        self.io.print_log('\tbest test RMSE loss: {}'.format(self.best_performance['RMSE']))
        self.io.print_log('\tbest test R2: {}'.format(self.best_performance['R2']))
        self.io.print_log('\tTop1 return ratio: {:.6f}'.format(self.best_performance['top1']))
        self.io.print_log('\tTop5 return ratio: {:.6f}'.format(self.best_performance['top5']))
        self.io.print_log('\tTop10 return ratio: {:.6f}'.format(self.best_performance['top10']))

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser])

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5, 10], nargs='+', help='which Top K return ration will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='Adam', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser

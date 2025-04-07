import argparse
import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr, pearsonr
# import matplotlib.pyplot as plt

def get_parser(data_type, adj_type):
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, default=15)
    if data_type == 'HKEXtoSZSE':
        parser.add_argument('--window_size', type=int, default=3,
                            help='计算节点相关性时取窗口的大小')
        parser.add_argument('--feature_index', type=str, default='data/HKEXtoSZSE/feature_to_ind.pkl',
                            help='dict of feature index.')
        parser.add_argument('--stock_filename', type=str, default='data/HKEXtoSZSE/Processed_HKEXtoSZSE.pkl',
                            help='File containing stock imfomations.')
        parser.add_argument('--normalized_k', type=float, default=0.1,
                            help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
        if adj_type == 'amount':
            parser.add_argument('--output_pkl_filename', type=str, default='data/adj_mx/dyadjmx_amount_HKEXtoSZSE_t=5.pkl',
                            help='Path of the output file.')
        else:
            parser.add_argument('--output_pkl_filename', type=str, default='data/adj_mx/dyadjmx_closeprice_HKEXtoSZSE.pkl',
                            help='Path of the output file.')
    elif data_type == 'HKEXtoSSE':
        parser.add_argument('--window_size', type=int, default=3,
                            help='计算节点相关性时取窗口的大小')
        # parser.add_argument('--feature_index', type=str, default='data/HKEXtoSSE/feature_to_ind.pkl',
        #                     help='dict of feature index.')
        # parser.add_argument('--stock_filename', type=str, default='data/HKEXtoSSE/Processed_HKEXtoSSE.pkl',
        #                     help='File containing stock imfomations.')
        parser.add_argument('--feature_index', type=str, default='data/casestudy/feature_to_ind.pkl',
                            help='dict of feature index.')
        parser.add_argument('--stock_filename', type=str, default='data/casestudy/Processed_HKEXtoSSE.pkl',
                            help='File containing stock imfomations.')

        parser.add_argument('--normalized_k', type=float, default=0.1,
                            help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
        if adj_type == 'amount':
            parser.add_argument('--output_pkl_filename', type=str, default='data/casestudy/dyadjmx_amount_HKEXtoSSE.pkl',
                            help='Path of the output file.')
        else:
            parser.add_argument('--output_pkl_filename', type=str, default='data/casestudy/dyadjmx_closeprice_HKEXtoSSE.pkl',
                            help='Path of the output file.')

    elif data_type == 'sp500':
        parser.add_argument('--window_size', type=int, default=3,
                            help='计算节点相关性时取窗口的大小')
        parser.add_argument('--feature_index', type=str, default='data/sp500/feature_to_ind.pkl',
                            help='dict of feature index.')
        parser.add_argument('--stock_filename', type=str, default='data/sp500/Processed_sp500b.pkl',
                            help='File containing stock imfomations.')

        parser.add_argument('--normalized_k', type=float, default=0.1,
                            help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
        if adj_type == 'amount':
            parser.add_argument('--output_pkl_filename', type=str, default='data/t=5/sp500/dyadjmx_amount_sp500_t=5.pkl',
                            help='Path of the output file.')
        else:
            parser.add_argument('--output_pkl_filename', type=str, default='data/sp500/dyadjmx_closeprice_sp500.pkl',
                            help='Path of the output file.')

    args = parser.parse_args()
    return args


def get_adjacency_matrix(stock_data, stock_ids, normalized_k, window_size, dynet):
    """
    :param stock_data: 时间-股票数据，ndarray.
    :param stock_ids: list of stock ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :param n_step: 计算节点相关性时取前n天的数据.
    :return:
    """
    num_stocks = len(stock_ids)
    rel_mx = np.zeros((num_stocks, num_stocks), dtype=np.float32)
    # 计算收盘价的差值
    # close_change = stock_data[:, :, 1][:, 1:] - stock_data[:, :, 1][:, :-1]
    # 计算成交额度的差值
    # amount_change = stock_data[:, :, 5][:, 1:] - stock_data[:, :, 5][:, :-1]
    amount_change = stock_data[:, :, 4][:, 1:] - stock_data[:, :, 4][:, :-1]


    # 试一下固定的矩阵
    # corr_matrix, _ = spearmanr(close_change, axis=1)
    corr_matrix, _ = spearmanr(amount_change, axis=1)
    # 相关性矩阵中小于矩阵均值的元素均为0，视为两个股票之间没有相关性
    normalized_k = np.mean(corr_matrix)
    corr_matrix[corr_matrix < normalized_k] = 0

    if dynet == False:    # 若得到静态网络
        with open(args.output_pkl_filename, 'wb') as f:
            pickle.dump(corr_matrix, f, protocol=2)
    else:
        corr_matrix_ndarray = np.zeros((amount_change.shape[1] - window_size + 1, amount_change.shape[0], amount_change.shape[0]))
        # 使用循环移动窗口
        for i in range(amount_change.shape[1] - window_size + 1):
            # 提取窗口数据
            window_data = amount_change[:, i:i + window_size]

            # 计算Spearman相关系数
            # 我们可以使用 window_data 计算每一对节点之间在该窗口期间的相关性
            corr_matrix_p = np.corrcoef(window_data)

            # 将所有NaN值替换为0
            np.nan_to_num(corr_matrix_p, copy=False, nan=0)
            # 将所有小于0的值设置为0
            # corr_matrix_p[corr_matrix_p < 0] = 0

            a = 0.5
            b = 0.5

            corr_matrix_total = a * corr_matrix_p + b * corr_matrix
            corr_matrix_total[corr_matrix_total < normalized_k] = 0
            # 存储最终的邻接矩阵（a*整体相关性矩阵+b*局部相关性矩阵）
            # 这里考虑是先把小于阈值的值改为0还是在相加后把小于阈值的值改为0
            # 存储到三维numpy数组的对应层
            corr_matrix_ndarray[i] = corr_matrix_total
            print('已经生成的矩阵数量：', i)

        corr_matrix_ndarray = corr_matrix_ndarray.astype(np.float32)

        # return corr_matrix_ndarray
        print("矩阵已经生成完毕，开始保存")
        # 保存邻接矩阵
        np.savez_compressed(args.output_pkl_filename, corr_matrix_ndarray)
        print('矩阵已保存')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = get_parser('sp500', 'amount')

    with open(args.stock_filename, 'rb') as f:
        stock_ids, stock_id_to_ind, stock_data = pickle.load(f)
    with open(args.feature_index, 'rb') as f:
        feature_to_ind = pickle.load(f)


    # get_adjacency_matrix(stock_data, stock_ids, normalized_k=0.1, window_size=3, dynet=False)
    # get_adjacency_matrix(stock_data, stock_ids, normalized_k=0.1, window_size=1, dynet=True)
    get_adjacency_matrix(stock_data, stock_ids, normalized_k=0.1, window_size=5, dynet=True)




import argparse
import os
import numpy as np
import pandas as pd
import pickle
from numpy.lib.stride_tricks import as_strided
# from sklearn.model_selection import train_test_split

def extract_time_windows(stock_data, adj, feature_to_ind, time_steps):
    """

    Args:
        stock_data: shape:(feature number, dates, nodes )
        adj: (dates, nodes, nodes)
        feature_to_ind:
        time_steps:

    Returns:
        output:(date,feature,timestep,nodes)
        closing_rice:(date, node)
        label:(date, node)
    """
    x_stock_data_indices = [feature_to_ind['Open'], feature_to_ind['High'], feature_to_ind['Low'], feature_to_ind['Volume']]
    x_stock_data = stock_data[x_stock_data_indices, :, :]
    x_stock_data = np.nan_to_num(x_stock_data, nan=0.0)
    closing_price = stock_data[feature_to_ind['Close'], :-time_steps, :]

    label = stock_data[feature_to_ind['ChangeRatio'], :-time_steps, :]
    label_new = []
    for i in range(label.shape[0]):
        label_new.append(label[i])

    num_features, num_dates, num_stocks = x_stock_data.shape
    num_windows = num_dates - time_steps
    for i in range(adj.shape[0]):
        adj[i] = normalize_undigraph(adj[i])
    num_dates_adj = adj.shape[0]
    num_windows_adj = num_dates_adj - time_steps
    # 初始化输出数组
    output = np.zeros((num_windows, num_features, time_steps, num_stocks))
    # out_adj = np.zeros((num_windows_adj, time_steps, num_stocks, num_stocks))

    for t in range(num_windows):
        output[t] = x_stock_data[:, t:t+time_steps, :]

    # 计算出切片的步长和形状
    adj_strides = adj.strides
    shape = (num_windows_adj, time_steps) + adj.shape[1:]
    out_adj = as_strided(adj, shape=shape, strides=(adj_strides[0], adj_strides[0], adj_strides[1], adj_strides[2]))

    return output, closing_price, label_new, out_adj

def split_data_by_ratio(x_stock_data, closing_price, label_new, adj_new, train_ratio, val_ratio, args):
    total_samples = adj_new.shape[0]

    # 计算每个数据集的索引范围
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)
    print(train_end,val_end)
    # 根据索引范围提取数据
    datasets = {
        'train': {
            'x': x_stock_data[1:train_end + 1],
            'adj': adj_new[:train_end],
            'closing': closing_price[1:train_end + 1],
            'label': label_new[1:train_end + 1]
        },
        'val': {
            'x': x_stock_data[train_end + 1:val_end + 1],
            'adj': adj_new[train_end:val_end],
            'closing': closing_price[train_end + 1:val_end + 1],
            'label': label_new[train_end + 1:val_end + 1]
        },
        'test': {
            'x': x_stock_data[val_end + 1:total_samples + 1],
            'adj': adj_new[val_end:],
            'closing': closing_price[val_end + 1:total_samples + 1],
            'label': label_new[val_end + 1:total_samples + 1]
        }
    }
    print('开始保存数据')
    # 保存数据
    for dataset_name, dataset in datasets.items():
        output_path = f"{args.output_dir}/{dataset_name}"
        np.save(f'{output_path}_15_EOD.npy', dataset['x'])
        np.save(f'{output_path}_15_price.npy', dataset['closing'])
        np.savez_compressed(f'{output_path}_15_adj', dataset['adj'])
        with open(f'{output_path}_15_label.pkl', 'wb') as f:
            pickle.dump(dataset['label'], f)

    # # 根据索引范围提取数据
    # x_train = x_stock_data[1:train_end+1]
    # x_val = x_stock_data[train_end+1:val_end+1]
    # x_test = x_stock_data[val_end+1:total_samples+1]
    #
    # adj_train = adj_new[:train_end]
    # adj_val = adj_new[train_end:val_end]
    # adj_test = adj_new[val_end:]
    #
    # closing_train = closing_price[1:train_end+1]
    # closing_val = closing_price[train_end+1:val_end+1]
    # closing_test = closing_price[val_end+1:total_samples+1]
    #
    # label_train = label_new[1:train_end+1]
    # label_val = label_new[train_end+1:val_end+1]
    # label_test = label_new[val_end+1:total_samples+1]
    #
    # np.save('train_15_EOD.npy', x_train)
    # np.save('val_15_EOD.npy', x_val)
    # np.save('test_15_EOD.npy', x_test)
    #
    # np.save('train_15_price.npy', closing_train)
    # np.save('val_15_price.npy', closing_val)
    # np.save('test_15_price.npy', closing_test)
    #
    # np.savez_compressed('adj_train_15', adj_train)
    # np.savez_compressed('adj_val_15', adj_val)
    # np.savez_compressed('adj_test_15', adj_test)
    # # # 保存为 .pkl 格式
    # with open('train_15_label.pkl', 'wb') as f:
    #     pickle.dump(label_train, f)
    #
    # with open('val_15_label.pkl', 'wb') as f:
    #     pickle.dump(label_val, f)
    #
    # with open('test_15_label.pkl', 'wb') as f:
    #     pickle.dump(label_test, f)

def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

def generate_train_val_test(args):

    with open(args.stock_filename, 'rb') as f:
        stock_ids, stock_id_to_ind, stock_data = pickle.load(f)

    with open(args.feature_index, 'rb') as f:
        feature_to_ind = pickle.load(f)

    adj = np.load(args.adj_filename)
    adj = adj['arr_0']

    stock_data = stock_data.transpose(2, 1, 0)

    x_stock_data, closing_price, label_new, adj_new = extract_time_windows(stock_data, adj, time_steps=args.time_steps, feature_to_ind=feature_to_ind)

    split_data_by_ratio(x_stock_data, closing_price, label_new, adj_new, args.train_ratio, args.val_ratio, args)

def get_parser(data_type):
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, default=15)
    if data_type == 'sp500':
        parser.add_argument('--feature_index', type=str, default='data/sp500/feature_to_ind.pkl',
                            help='dict of feature index.')
        parser.add_argument('--stock_filename', type=str, default='data/sp500/Processed_sp500b.pkl',
                            help='File containing stock imfomations.')
        parser.add_argument('--train_ratio', type=float, default=0.7)
        parser.add_argument('--val_ratio', type=float, default=0.1)
        # ============================================================================================================
        parser.add_argument("--output_dir", type=str, default="data/sp500", help="Output directory.")
        parser.add_argument("--adj_filename", type=str, default="data/sp500/dyadjmx_amount_sp500.pkl.npz")
        # ============================================================================================================
        # parser.add_argument("--output_dir", type=str, default="data/t=1/sp500", help="Output directory.")
        # parser.add_argument("--adj_filename", type=str, default="data/t=1/sp500/dyadjmx_amount_sp500_t=1.pkl.npz")
        # ============================================================================================================
        # parser.add_argument("--output_dir", type=str, default="data/t=5/sp500", help="Output directory.")
        # parser.add_argument("--adj_filename", type=str, default="data/t=5/sp500/dyadjmx_amount_sp500_t=5.pkl.npz")

    elif data_type == 'HKEXtoSZSE':
        parser.add_argument('--feature_index', type=str, default='data/HKEXtoSZSE/feature_to_ind.pkl',
                            help='dict of feature index.')
        parser.add_argument('--stock_filename', type=str, default='data/HKEXtoSZSE/Processed_HKEXtoSZSE.pkl',
                            help='File containing stock imfomations.')
        parser.add_argument('--train_ratio', type=float, default=0.7)
        parser.add_argument('--val_ratio', type=float, default=0.1)
        # parser.add_argument("--output_dir", type=str, default="data/HKEXtoSZSE", help="Output directory.")
        # parser.add_argument("--adj_filename", type=str, default="data/adj_mx/dyadjmx_amount_HKEXtoSZSE.pkl.npz")
        # ============================================================================================================
        # parser.add_argument("--output_dir", type=str, default="data/t=1/SZSE", help="Output directory.")
        # parser.add_argument("--adj_filename", type=str, default="data/t=1/SZSE/dyadjmx_amount_HKEXtoSZSE_t=1.pkl.npz")
        # ============================================================================================================
        parser.add_argument("--output_dir", type=str, default="data/t=5/SZSE", help="Output directory.")
        parser.add_argument("--adj_filename", type=str, default="data/t=5/SZSE/dyadjmx_amount_HKEXtoSZSE_t=5.pkl.npz")


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser('sp500')

    generate_train_val_test(args)

import numpy as np
import os
import matplotlib.pyplot as plt


def load_dataset(prefix, data_dir='.'):
    """
    根据输入前缀读取对应数据集（o/b/c 后缀），并提取 result/cp 数组

    参数:
        prefix (str): 数据集前缀，如 'case', 'sse', 'szse', 'sp500'
        data_dir (str): 数据文件目录，默认为当前目录

    返回:
        dict: 结构为 { 'o': {'result': arr, 'cp': arr}, 'b': {...}, 'c': {...} }

    异常:
        FileNotFoundError: 文件不存在
        KeyError: 文件缺失 result 或 cp 数组
    """
    suffixes = ['o', 'b', 'c', 'g']
    # suffixes = ['o', 'b', 'c']
    dataset = {}

    for suffix in suffixes:
        file_name = f"{prefix}-{suffix}.npz"
        file_path = os.path.join(data_dir, file_name)

        # 加载文件并提取数据
        try:
            npz_file = np.load(file_path)
            dataset[suffix] = {
                'result': npz_file['result'],  # 直接提取 result 数组
                'cp': npz_file['cp']  # 直接提取 cp 数组
            }
        except FileNotFoundError:
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        except KeyError as e:
            raise KeyError(f"文件 {file_path} 中缺失必要字段: {str(e)}")
        finally:
            if 'npz_file' in locals():  # 确保文件句柄关闭
                npz_file.close()

    return dataset


if __name__ == '__main__':

    data = load_dataset('sse')
    o_result = data['o']['result']
    c_result = data['c']['result']
    b_result = data['b']['result']
    g_result = data['g']['result'] #case 注释掉
    cp = data['o']['cp']

    row_means1 = np.mean(o_result, axis=1)
    row_means2 = np.mean(c_result, axis=1)
    row_means3 = np.mean(b_result, axis=1)
    row_means5 = np.mean(g_result, axis=1) #case 注释掉
    row_means4 = np.mean(cp, axis=1)

    # 绘制整体市场均值的图
    # # 假设 row_means1 和 row_means2 是两个长度相同的一维数组
    # x = range(len(row_means1[:60]))
    #
    # # 创建一个新的图形和轴
    # fig, ax = plt.subplots(figsize=(12, 4))
    #
    # # 绘制折线图
    # ax.plot(x, row_means4[:60], '#1f77b4', label='Ground Truth', alpha=0.7, linewidth=0.8)
    # ax.plot(x, row_means5[:60],'g', label='GRU', alpha=0.7, linewidth=0.8)
    # ax.plot(x, row_means1[:60], 'r', label='DyCGCN(O)', alpha=0.8, linewidth=0.7)
    # # ax.plot(x, row_means3, label='Balanced Strategy', alpha=0.7, linewidth=0.8, marker='o', markersize=0.1)

    # 绘制单只股票的图
    # 假设 row_means1 和 row_means2 是两个长度相同的一维数组
    x = range(len(row_means1[:60]))

    # 创建一个新的图形和轴
    fig, ax = plt.subplots(figsize=(12, 4))

    # 绘制折线图
    ax.plot(x, cp[:,1][:60], '#1f77b4', label='Ground Truth', alpha=0.7, linewidth=0.8)
    ax.plot(x, g_result[:,1][:60],'g', label='GRU', alpha=0.7, linewidth=0.8)
    ax.plot(x, o_result[:,1][:60], 'r', label='DyCGCN(O)', alpha=0.8, linewidth=0.7)
    # ax.plot(x, row_means3, label='Balanced Strategy', alpha=0.7, linewidth=0.8, marker='o', markersize=0.1)


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
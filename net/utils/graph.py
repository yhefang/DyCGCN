import numpy as np
import pickle

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 market,
                 strategy='uniform',
                 relation_path=None):
        # 改成适应于的动态网络的情况
        self.get_edge(market, relation_path)
        self.get_adjacency()

    def __str__(self):
        return self.A

    def get_edge(self, market, relation_path):
        if market == 'HKEXtoSSE':
            self.num_node = 896
            # self.num_node = 594 # for case study
            if relation_path.endswith('.npz'):
                with np.load(relation_path, allow_pickle=True) as data:
                    array_key = data.files[0]
                    # 提取数组
                    raw_relation = data[array_key]
                self.relation = raw_relation
                self.edge = raw_relation
            else:
                with open(relation_path, 'rb') as f:
                    raw_relation = pickle.load(f)
                self.relation = raw_relation
                self.edge = raw_relation

        elif market == 'HKEXtoSZSE':
            self.num_node = 1205
            if relation_path.endswith('.npz'):
                with np.load(relation_path, allow_pickle=True) as data:
                    array_key = data.files[0]
                    # 提取数组
                    raw_relation = data[array_key]
                self.relation = raw_relation
                self.edge = raw_relation
            else:
                with open(relation_path, 'rb') as f:
                    raw_relation = pickle.load(f)
                self.relation = raw_relation
                self.edge = raw_relation

        elif market == 'NASDAQ':
            self.num_node = 467
            if relation_path.endswith('.npz'):
                with np.load(relation_path, allow_pickle=True) as data:
                    array_key = data.files[0]
                    # 提取数组
                    raw_relation = data[array_key]
                self.relation = raw_relation
                self.edge = raw_relation
            else:
                with open(relation_path, 'rb') as f:
                    raw_relation = pickle.load(f)
                self.relation = raw_relation
                self.edge = raw_relation

        elif market == 'sp500':
            self.num_node = 469
            if relation_path.endswith('.npz'):
                with np.load(relation_path, allow_pickle=True) as data:
                    array_key = data.files[0]
                    # 提取数组
                    raw_relation = data[array_key]
                self.relation = raw_relation
                self.edge = raw_relation
            else:
                with open(relation_path, 'rb') as f:
                    raw_relation = pickle.load(f)
                self.relation = raw_relation
                self.edge = raw_relation

        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self):
        adjacency = self.edge

        # 动态邻接矩阵
        if len(adjacency.shape) >= 3:
            Ai = np.zeros((adjacency.shape[0], self.num_node, self.num_node))
            for i in range(adjacency.shape[0]):
                normalize_adjacency = normalize_undigraph(adjacency[i])
                Ai[i] = normalize_adjacency
            self.A = Ai

        elif len(adjacency.shape) < 3:
            # A = np.zeros((self.num_node, self.num_node))
            normalize_adjacency = normalize_undigraph(adjacency)
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            # A[0] = normalize_adjacency
            self.A = A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

if __name__ == '__main__':
    graph = Graph(market='HKEXtoSSE',
                 strategy='uniform',
                 relation_path='../../data/HKEXtoSSE/adjmx_amount_HKEXtoSSE.pkl')
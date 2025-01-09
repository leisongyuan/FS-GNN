# import required modules
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import nn, Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

'''模拟删特征'''
# 掩码mask用于表示哪些特征数据应该被标记为缺失值
def feature_mask(features, missing_rate):
    mask = torch.rand(size=features.size())
    # 这一行将掩码中小于等于 missing_rate 的值转换为 True，大于 missing_rate 的值转换为 False。
    mask = mask <= missing_rate
    return mask
# 将相应位置的特征数据标记为缺失值
def apply_feature_mask(features, mask):
    features[mask] = float('0')

'''构建属性相似边'''
def feature_sim_edges(feature_matrix, topk):

    feature_matrix = feature_matrix.detach().cpu()
    # 计算余弦相似度
    cosine_sim = cosine_similarity(feature_matrix)

    # 构建图的边
    edges = []
    for i in range(len(cosine_sim)):
        # 对于每个节点，找到最相似的topk个节点
        topk_indices = np.argsort(-cosine_sim[i])[:topk + 1]  # 包含自身，所以是 topk+1
        for j in topk_indices:
            if i != j and cosine_sim[i][j] > 0.8:  # 排除自身且相似度大于0.7
                edges.append((i, j, cosine_sim[i][j]))
    
    return edges


'''构建属性相似图'''
def feature_sim_graph(emb, topk):
    # 构建属性相似图，补同类型节点之间的边
    rows = []
    cols = []

    sim_edges = feature_sim_edges(emb, topk)
    for edge in sim_edges:
        rows = np.append(rows, edge[0])
        cols = np.append(cols, edge[1])

    # 将 rows 和 cols 转换为 PyTorch tensors
    rows_tensor = torch.from_numpy(np.array(rows)).long()
    cols_tensor = torch.from_numpy(np.array(cols)).long()

    fea_sim_edge_index = torch.stack([rows_tensor, cols_tensor], dim=0)

    return fea_sim_edge_index
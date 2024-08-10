import numpy as np
import json
import matplotlib.pyplot as plt
import networkx as nx
from dgl.nn.pytorch import GraphConv
from model.model import GCN
# from node2vec import Node2Vec
from utils.create_graph_dgl_utils import create_graph


# 参数
emb_dim = 64
topk = 5
layer_num = 2


def createDataset(cm_array, topk):
    gt_list = []
    for i in range(cm_array.shape[0]):
        non_zero_columns = np.nonzero(cm_array[i])[0].tolist()
        gt_list.append([i, non_zero_columns])
    return gt_list


def train(cm_array, layer_num):
    n_nodes = cm_array.shape[0]
    gcn = GCN(n_nodes, emb_dim, layer_num)


def main():
    # 读取互补特征
    features = np.load('../dataset/processed/comp_feature/entropy_c_features.npy').tolist()
    print('File Read Finished')

    embeddings_dict = {}
    for feature in features:
        # 读取feature_value列表和共现矩阵
        cm_array = np.load(f'../dataset/processed/cm_fv/{feature}_cm.npy')

        graph = create_graph(cm_array)  # 根据共现矩阵造图
        nx_graph = graph.to_networkx().to_undirected()  # 将DGL图对象转换为NetworkX图对象

        # 构造训练数据

        # 训练


        # # 初始化并训练node2vec
        # node2vec = Node2Vec(nx_graph, dimensions=emb_dim, walk_length=5, num_walks=100, workers=4, p=1, q=0.25, weight_key='weight')
        # model = node2vec.fit(window=5, min_count=1, batch_words=4)

        embeddings = model.wv.vectors  # 获取embedding
        embeddings_dict[feature] = embeddings.tolist()  # 保存数据到字典
        print(f'{feature} Finished')
    # 将字典数据保存到JSON文件中
    with open("../dataset/processed/embeddings/embeddings.json", "w") as json_file:
        json.dump(embeddings_dict, json_file)


if __name__ == '__main__':
    main()










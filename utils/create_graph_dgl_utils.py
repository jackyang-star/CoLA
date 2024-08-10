import os
import dgl
import torch
os.environ["DGLBACKEND"] = "pytorch"


# 构建feature_values的共现图
def create_graph(cm):
    # 创建一个空的DGL图
    graph = dgl.DGLGraph()

    # 添加节点，节点的数量为feature_value的数量
    num_nodes = cm.shape[0]
    graph.add_nodes(num_nodes)

    # 遍历共现矩阵，添加边
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if cm[i, j] > 0:
                graph.add_edges(i, j, data={'weight': torch.tensor([cm[i, j]])})
                graph.add_edges(j, i, data={'weight': torch.tensor([cm[i, j]])})

    return graph









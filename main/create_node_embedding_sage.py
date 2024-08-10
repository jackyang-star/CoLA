import dgl
import torch
import numpy as np
import scipy.sparse as sp
import json
from model.model import GraphSAGE


# 将邻接矩阵转换为DGL图的函数
def create_dgl_graph(adj_matrix):
    sp_matrix = sp.coo_matrix(adj_matrix)
    g = dgl.from_scipy(sp_matrix)
    return g

# 邻接矩阵
adj_matrices = []
graph_names = []
features = np.load('../dataset/processed/comp_feature/entropy_c_features.npy').tolist()
for feature in features:
    # 读取feature_value列表和共现矩阵
    cm_array = np.load(f'../dataset/processed/cm_fv/{feature}_cm.npy')
    adj_matrices.append(cm_array)
    graph_names.append(feature)

# 将所有邻接矩阵转换为DGL图
graphs = [create_dgl_graph(adj) for adj in adj_matrices]


# 计算损失函数
def compute_loss(pos_score, neg_score):
    loss = -torch.log(pos_score.sigmoid() + 1e-15).mean() - torch.log(1 - neg_score.sigmoid() + 1e-15).mean()
    return loss

# 负采样函数
def edge_sampler(g, num_neg_samples):
    u, v = g.edges()
    neg_src = u.repeat(num_neg_samples)
    neg_dst = torch.randint(0, g.number_of_nodes(), (len(u) * num_neg_samples,))
    return (neg_src, neg_dst)

# 训练GraphSAGE模型，增加早停法
def train(g, num_nodes, feats_dim, num_epochs=500, patience=20):
    model = GraphSAGE(num_nodes, feats_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    node_ids = torch.arange(num_nodes)
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        h = model(g, node_ids)
        pos_score = (h[g.edges()[0]] * h[g.edges()[1]]).sum(dim=1)

        neg_src, neg_dst = edge_sampler(g, num_neg_samples=20)
        neg_score = (h[neg_src] * h[neg_dst]).sum(dim=1)

        loss = compute_loss(pos_score, neg_score)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')

        # 早停法
        if loss.item() < best_loss:
            best_loss = loss.item()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    return model

# 获取节点嵌入
def get_node_embeddings(model, g, num_nodes):
    node_ids = torch.arange(num_nodes)
    with torch.no_grad():
        h = model(g, node_ids)
    return h

# 设置参数
feats_dim = 64  # 初始嵌入和输出嵌入的维度

# 训练每个图并获取节点嵌入
embeddings_dict = {}
for i, (g, name) in enumerate(zip(graphs, graph_names)):
    num_nodes = g.number_of_nodes()  # 动态获取每个图的节点数
    model = train(g, num_nodes, feats_dim)
    emb = get_node_embeddings(model, g, num_nodes)
    embeddings_dict[name] = emb.tolist()  # 将tensor转换为列表以便序列化为JSON


# 将嵌入字典保存到JSON文件
with open('../dataset/processed/embeddings/node_embeddings.json', 'w') as f:
    json.dump(embeddings_dict, f)

print("Node embeddings have been saved to 'node_embeddings.json'")










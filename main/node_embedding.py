import pandas as pd
import numpy as np
import dgl
import torch
import scipy.sparse as sp
import json
from model.model import GraphSAGE
from utils.create_cm_fv_utils import create_fv_list, create_api_cooccurrence_matrix, create_fv_cooccurrence_matrix


# 参数
feats_dim = 64


def create_api_cm(api_df, mashup_df):
    # 获取API的列表和共现矩阵
    api_list = sorted(api_df['Name'].to_list())
    api_cm_array = create_api_cooccurrence_matrix(mashup_df, api_list, 'Related APIs')
    # 删除没有共现关系的API
    non_zero_rows = np.any(api_cm_array != 0, axis=1)
    non_zero_cols = np.any(api_cm_array != 0, axis=0)
    non_zero_row_indices = np.nonzero(non_zero_rows)[0]
    api_list = [api_list[i] for i in non_zero_row_indices]
    api_list_array = np.array(api_list)
    api_cm_array = api_cm_array[non_zero_rows][:, non_zero_cols]
    # # 删除没有互补特征的API(使用noname_features时启用)
    # delete_index = []
    # for api in api_list:
    #     index = api_list.index(api)
    #     count = 0
    #     for feature in features:
    #         matching_api = api_df[api_df['Name'] == api]
    #         matching_feature = matching_api[feature].unique()
    #         if pd.isna(matching_feature):
    #             count += 1
    #     if count == len(features):
    #         delete_index.append(index)
    # api_list_array = np.delete(api_list, delete_index)
    # api_cm_array = np.delete(api_cm_array, delete_index, axis=0)
    # api_cm_array = np.delete(api_cm_array, delete_index, axis=1)
    # # 再次删除没有共现关系的API
    # non_zero_rows = np.any(api_cm_array != 0, axis=1)
    # non_zero_cols = np.any(api_cm_array != 0, axis=0)
    # non_zero_row_indices = np.nonzero(non_zero_rows)[0]
    # api_list_array = api_list_array[non_zero_row_indices]
    # api_cm_array = api_cm_array[non_zero_rows][:, non_zero_cols]

    return api_list_array, api_cm_array


# 将邻接矩阵转换为DGL图的函数
def create_dgl_weighted_graph(adj_matrix):
    # 确保权重为 int32 类型
    adj_matrix = adj_matrix.astype(np.int32)
    sp_matrix = sp.coo_matrix(adj_matrix)
    g = dgl.from_scipy(sp_matrix, eweight_name='weight')
    g.edata['weight'] = g.edata['weight'].type(torch.int32)
    return g


def create_dglgraphs(features, api_df, api_list_array, api_cm_array):
    max_length = 0
    adj_matrices = []
    graph_names = []
    for feature in features:
        fv_list_array = create_fv_list(api_df, api_list_array, feature)  # 获取feature_value列表
        cm_array, max_fv_len = create_fv_cooccurrence_matrix(api_cm_array, api_list_array, api_df, fv_list_array,
                                                             feature)  # 构建feature_value共现矩阵
        adj_matrices.append(cm_array)
        graph_names.append(feature)
        if max_fv_len > max_length:
            max_length = max_fv_len  # 获取特征值最大数量

    graphs = [create_dgl_weighted_graph(adj) for adj in adj_matrices]
    np.savetxt(f'../dataset/processed/cm_fv/max_fv_length.txt', np.array([max_length]), fmt='%d')  # 保存特征值最大数量

    return graph_names, graphs


# 计算损失函数
def compute_loss(pos_score, neg_score):
    loss = -torch.log(pos_score.sigmoid() + 1e-15).mean() - torch.log(1 - neg_score.sigmoid() + 1e-15).mean()
    return loss


# 负采样函数
def edge_sampler(g, num_neg_samples):
    u, v = g.edges()
    edge_set = set(zip(u.numpy(), v.numpy()))  # 将正样本转换为集合，便于快速查找
    neg_src = []
    neg_dst = []
    while len(neg_src) < len(u) * num_neg_samples:
        src = u.repeat(num_neg_samples)
        dst = torch.randint(0, g.number_of_nodes(), (len(u) * num_neg_samples,))

        for i in range(len(src)):
            if (src[i].item(), dst[i].item()) not in edge_set and (dst[i].item(), src[i].item()) not in edge_set:
                neg_src.append(src[i])
                neg_dst.append(dst[i])
    # 将列表转换为张量
    neg_src = torch.stack(neg_src[:len(u) * num_neg_samples])
    neg_dst = torch.stack(neg_dst[:len(u) * num_neg_samples])

    return neg_src, neg_dst


# 对比学习，训练模型，增加早停法
def train(g, num_nodes, feats_dim, num_epochs=1000, patience=20):
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


def main():
    # 读入文件
    api_df = pd.read_json('../dataset/raw/programmableweb/apiData.json')
    mashup_df = pd.read_json('../dataset/raw/programmableweb/mashupData.json')
    features = np.load('../dataset/processed/comp_feature/entropy_c_features.npy').tolist()
    # noname_features = np.load('../dataset/processed/comp_feature/noname_entropy_c_features.npy').tolist()
    # noother_features = np.load('../dataset/processed/comp_feature/noother_entropy_c_features.npy').tolist()

    # 生成dgl图
    api_list_array, api_cm_array = create_api_cm(api_df, mashup_df)
    graph_names, graphs = create_dglgraphs(features, api_df, api_list_array, api_cm_array)

    # 训练得到node embedding
    embeddings_dict = {}
    for i, (g, name) in enumerate(zip(graphs, graph_names)):
        num_nodes = g.number_of_nodes()  # 动态获取每个图的节点数
        model = train(g, num_nodes, feats_dim)
        emb = get_node_embeddings(model, g, num_nodes)
        embeddings_dict[name] = emb.tolist()  # 将tensor转换为列表以便序列化为JSON

    # 将嵌入字典保存到JSON文件
    with open('../dataset/processed/embeddings/node_embeddings.json', 'w') as f:
        json.dump(embeddings_dict, f)

if __name__ == '__main__':
    main()










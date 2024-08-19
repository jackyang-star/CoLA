import numpy as np
import pandas as pd
import dgl
import torch
from torch import nn
import scipy.sparse as sp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from model.model import EarlyStopping
from model.model2 import GraphSAGE, Predictor, MVCG
from utils.create_cm_fv_utils import create_fv_list, create_api_cooccurrence_matrix, create_fv_cooccurrence_matrix


# 参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
embedding_dim = 64
h_dim = 16
num_epochs = 1000
patience = 25
delta = 0
ifverbose = False


# 构建API的共现矩阵
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


# 将不同特征的共现矩阵转换为DGL图
def create_dgl_weighted_graph(adj_matrix):
    # 确保权重为 int32 类型
    adj_matrix = adj_matrix.astype(np.int32)
    sp_matrix = sp.coo_matrix(adj_matrix)
    g = dgl.from_scipy(sp_matrix, eweight_name='weight')
    g.edata['weight'] = g.edata['weight'].type(torch.int32)
    return g

def create_graphs(features, api_df, api_list_array, api_cm_array):
    # max_length = 0
    # adj_matrices = []
    # graph_names = []
    graphs = []
    for feature in features:
        fv_list_array = create_fv_list(api_df, api_list_array, feature)  # 获取feature_value列表
        cm_array, max_fv_len = create_fv_cooccurrence_matrix(api_cm_array, api_list_array, api_df, fv_list_array,
                                                             feature)  # 构建feature_value共现矩阵
        graphs.append(create_dgl_weighted_graph(cm_array))
        # adj_matrices.append(cm_array)
        # graph_names.append(feature)
        # if max_fv_len > max_length:
        #     max_length = max_fv_len  # 获取特征值最大数量
    # graphs = [create_dgl_weighted_graph(adj) for adj in adj_matrices]
    # np.savetxt(f'../dataset/processed/cm_fv/max_fv_length.txt', np.array([max_length]), fmt='%d')  # 保存特征值最大数量
    return graphs


def prepare_data(mvcg, graphs, node_embeddings):

    return train_pairs, labels


# 读入文件
api_df = pd.read_json('../dataset/raw/programmableweb/apiData.json')
mashup_df = pd.read_json('../dataset/raw/programmableweb/mashupData.json')
features = np.load('../dataset/processed/comp_feature/entropy_c_features.npy').tolist()
# noname_features = np.load('../dataset/processed/comp_feature/noname_entropy_c_features.npy').tolist()
# noother_features = np.load('../dataset/processed/comp_feature/noother_entropy_c_features.npy').tolist()

# 生成dgl图
api_list_array, api_cm_array = create_api_cm(api_df, mashup_df)
graphs = create_graphs(features, api_df, api_list_array, api_cm_array)

# # 准备数据
# combined_embeddings, train_pairs, labels = prepare_data(graphs)
# dataset = PairDataset(combined_embeddings, train_pairs, labels)
# test_size = int(len(dataset) * test_split)
# train_size = len(dataset) - test_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型
node_nums = []
for g in graphs:
    node_nums.append(g.number_of_nodes())
gnn_models = [GraphSAGE(in_feats=embedding_dim, h_feats=embedding_dim).to(device) for _ in range(len(graphs))]
combiner = Combiner().to(device)
predictor = Predictor(embedding_dim, h_dim, len(graphs)).to(device)
mvcg = MVCG(gnn_models, combiner, predictor, embedding_dim, node_nums).to(device)

# 训练&测试模型
optimizer = torch.optim.Adam(mvcg.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.5, patience=15)
early_stopping = EarlyStopping(patience, delta, ifverbose, path='../model/checkpoint.pt')
loss_fn = nn.BCEWithLogitsLoss().to(device)  # 用于二分类的交叉熵损失
train_epochs_loss = []
for epoch in range(num_epochs):
    # 训练
    print(f'Training begins. Running on {device}.')
    mvcg.train()
    train_epoch_loss = []
    for query, candidate, label in train_loader:
        # cuda上运行
        query = query.to(device)
        candidate = candidate.to(device)
        label = label.to(device)
        # 计算complementary score
        complementary_score = mvcg(graphs, query, candidate)
        # 计算损失
        loss = loss_fn(complementary_score, label)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录loss
        train_epoch_loss.append(loss.item())
    train_epochs_loss.append(np.average(train_epoch_loss))
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_epochs_loss[epoch]}')

    # 测试
    print(f'Testing begins. Running on {device}.')
    mvcg.eval()
    with torch.no_grad():
        pass

    # 调整学习率
    scheduler.step(train_epoch_loss[-1])

    # 早停
    early_stopping(train_epochs_loss[-1])
    if early_stopping.early_stop:
        print("Early stopping")
        break










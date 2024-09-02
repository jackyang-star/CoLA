import datetime
import os
import sys
import numpy as np
import pandas as pd
import random
import dgl
from tqdm import tqdm
import torch
from torch import nn
import scipy.sparse as sp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from model.model2 import MVCG, EarlyStopping, MyDataset, Predictor, TeeOutput
from utils.create_cm_fv_utils import create_fv_list, create_api_cooccurrence_matrix, create_fv_cooccurrence_matrix


# parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# data parameters
embedding_dim = 64
train_test_ratio = 0.8
# model parameters
dropout_rate = 0.2
combine_layer_num = 1
gcn_layer_num = 2
# train & test parameters
batch_size = 64
learning_rate = 0.001
num_epochs = 100
topk = 10
# early_stopping parameters
es_patience = 20
es_delta = 0
es_verbose = False
# scheduler parameters
s_patience = 10
s_verbose = True
s_factor = 0.1
s_mode = 'min'


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
    g.edata['weight'] = g.edata['weight'].type(torch.float32)
    return g.to(device)

def create_graphs(features, api_df, api_list_array, api_cm_array):
    max_feature_length = 0
    graphs = []
    for feature in features:
        fv_list_array = create_fv_list(api_df, api_list_array, feature)  # 获取feature_value列表
        cm_array, max_fv_len = create_fv_cooccurrence_matrix(api_cm_array, api_list_array, api_df, fv_list_array,
                                                             feature)  # 构建feature_value共现矩阵
        graphs.append(create_dgl_weighted_graph(cm_array))
        if max_fv_len > max_feature_length:
            max_feature_length = max_fv_len  # 获取特征值最大数量
    return graphs, max_feature_length


# train_data(query, pos, label; query, neg, label); test_data(query, candidates, truths)
def prepare_data(api_cm_array, api_list_array, api_df, features):
    train_data = []
    test_data = []
    strategies = []

    fv_dict = {}
    for feature in features:
        fv_dict[feature] = create_fv_list(api_df, api_list_array, feature)

    for i in range(api_cm_array.shape[0]):
        non_zero_columns = np.nonzero(api_cm_array[i])[0].tolist()
        zero_columns = np.where(api_cm_array[i] == 0)[0].tolist()
        train_length = int(len(non_zero_columns) * train_test_ratio)
        # 构建训练数据
        positives = non_zero_columns[0:train_length]
        negatives = [col for col in zero_columns if col != i]  # 负样本要排除自身
        negatives = random.sample(negatives, min(train_length, len(negatives)))
        for pos in positives:
            train_data.append((i, pos, 1))
        for neg in negatives:
            train_data.append((i, neg, 0))
        # 构建测试数据
        truths = non_zero_columns[train_length:]
        all = list(range(api_cm_array.shape[1]))
        candidates = list(set(all) - set(positives) - {i})
        test_data.append(([i], candidates, truths))

        api_name = api_list_array[i]  # 根据行号，从api_list中找到api的名称
        matching_api = api_df[api_df['Name'] == api_name]  # 根据api_name，从api_df中找到对应的行
        item = {}
        for index, feature in enumerate(features):
            matching_feature = matching_api[feature].iloc[0]  # 获取特征值
            if not pd.isna(matching_feature):
                indexes = []
                matching_feature = [e.strip() for e in matching_feature.split(',')]  # 特征值保存为列表形式
                for fv in matching_feature:
                    fv_index = np.where(fv_dict[feature] == fv)[0][0]  # 通过特征值的名称在特征值列表中找到特征值的编号
                    indexes.append(fv_index)
                item[index] = indexes
        strategies.append(item)

    return train_data, test_data, strategies


def train(dataloader, model, predictor, optimizer, loss_fn, max_feature_length, graphs):
    model.train()
    epoch_loss = []
    for batch_query, batch_candidate, batch_label in dataloader:
        # cuda上运行
        batch_query = batch_query.to(device)
        batch_candidate = batch_candidate.to(device)
        batch_label = batch_label.to(torch.float32).to(device)
        # 计算complementary score
        batch_query_embedding = model(graphs, batch_query, max_feature_length)
        batch_candidate_embedding = model(graphs, batch_candidate, max_feature_length)
        complementary_scores = predictor(batch_query_embedding, batch_candidate_embedding)
        # complementary_scores = model(graphs, batch_query, batch_candidate, max_feature_length)
        # 计算损失
        loss = loss_fn(complementary_scores, batch_label)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录loss
        epoch_loss.append(loss.item())
    return epoch_loss


def recall(pred, truth):
    hit_num = len(set(pred) & set(truth))
    recall = hit_num / len(truth)
    return recall


def mrr(pred, truth):
    reciprocal_rank = 0
    for rank, item in enumerate(pred, start=1):
        if item in truth:
            reciprocal_rank = 1 / rank
            break
    return reciprocal_rank


def dcg(r):
    r = np.asfarray(r)
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0
def ndcg(pred, truth):
    ideal = [1] * len(set(truth) & set(pred))
    actual = [1 if item in truth else 0 for item in pred]
    idcg = dcg(ideal)
    adcg = dcg(actual)
    ndcg = adcg / idcg if idcg > 0 else 0
    return ndcg


def test(dataloader, model, predictor, graphs, max_feature_length, topk):
    model.eval()
    with torch.no_grad():
        recall_value = []
        mrr_value = []
        ndcg_value = []
        item_embeddings = {}
        for i in range(len(dataloader)):
            index = torch.tensor(i).to(device)
            item_embeddings[i] = model(graphs, index, max_feature_length)

        for query, candidates, truths in tqdm(dataloader):
            # query = torch.tensor(query, dtype=torch.int32).to(device)
            # candidates = torch.tensor(candidates, dtype=torch.int32).to(device)
            truths = torch.tensor(truths, dtype=torch.int32).to(device)
            query_embedding = item_embeddings[query[0].item()]
            # 计算得分
            scores = []  # 保存candidate编号和score
            for candidate in candidates:
                candidate_embedding = item_embeddings[candidate.item()]
                score = predictor(query_embedding, candidate_embedding)
                # score = model(graphs, query, candidate, max_feature_length)
                scores.append((candidate, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            topk_candidates = [x[0].item() for x in scores[:topk]]
            # 计算指标
            truths = truths.tolist()
            recall_value.append(recall(topk_candidates, truths))
            mrr_value.append(mrr(topk_candidates, truths))
            ndcg_value.append(ndcg(topk_candidates, truths))
        avg_recall = sum(recall_value) / len(recall_value)
        avg_mrr = sum(mrr_value) / len(mrr_value)
        avg_ndcg = sum(ndcg_value) / len(ndcg_value)
    return avg_recall, avg_mrr, avg_ndcg


def log_to_file_with_terminal_output(log_dir=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 获取当前时间并生成文件名
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"log_{current_time}.txt"

            # 如果指定了log_dir，则将文件路径设置为log_dir，否则使用当前目录
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                log_filepath = os.path.join(log_dir, log_filename)
            else:
                log_filepath = log_filename

            # 打开日志文件
            with open(log_filepath, "w") as log_file:
                # 创建TeeOutput对象，同时写入文件和终端
                tee = TeeOutput(sys.stdout, log_file)

                # 替换sys.stdout
                original_stdout = sys.stdout
                sys.stdout = tee

                try:
                    # 执行传入的函数
                    func(*args, **kwargs)
                finally:
                    # 恢复sys.stdout
                    sys.stdout = original_stdout
                    print(f"Logs saved to {log_filepath}")

        return wrapper
    return decorator


@log_to_file_with_terminal_output(log_dir="../log")
def main():
    # 读入文件
    api_df = pd.read_json('../dataset/raw/programmableweb/apiData.json')
    mashup_df = pd.read_json('../dataset/raw/programmableweb/mashupData.json')
    features = np.load('../dataset/processed/comp_feature/entropy_c_features.npy').tolist()
    # noname_features = np.load('../dataset/processed/comp_feature/noname_entropy_c_features.npy').tolist()
    # noother_features = np.load('../dataset/processed/comp_feature/noother_entropy_c_features.npy').tolist()

    # 生成dgl图
    api_list_array, api_cm_array = create_api_cm(api_df, mashup_df)
    graphs, max_feature_length = create_graphs(features, api_df, api_list_array, api_cm_array)

    # 准备数据
    train_data, test_data, strategies = prepare_data(api_cm_array, api_list_array, api_df, features)
    train_dataset = MyDataset(train_data)
    test_dataset = MyDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # 初始化模型
    node_nums = []
    for g in graphs:
        node_nums.append(g.number_of_nodes())
    predictor = Predictor(embedding_dim).to(device)
    mvcg = MVCG(dropout_rate, embedding_dim, node_nums, strategies, device, combine_layer_num, gcn_layer_num).to(device)
    # print(mvcg)

    # 实例化
    optimizer = torch.optim.Adam(list(mvcg.parameters()) + list(predictor.parameters()), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode=s_mode, factor=s_factor, patience=s_patience, verbose=s_verbose)
    early_stopping = EarlyStopping(es_patience, es_delta, es_verbose, path='../model/checkpoint.pt')
    loss_fn = nn.BCEWithLogitsLoss().to(device)  # 用于二分类的交叉熵损失

    # 训练&测试模型
    maxRecall = 0.0
    maxMRR = 0.0
    maxNDCG = 0.0
    print(f'embedding_dim = {embedding_dim}')
    print(f'dropout_rate = {dropout_rate}')
    print(f'combine_layer_num = {combine_layer_num}')
    print(f'gcn_layer_num = {gcn_layer_num}')
    print(f'batch_size = {batch_size}\n')
    print(f'Running on {device}.')
    for epoch in range(num_epochs):
        # 训练
        epoch_loss = train(train_dataloader, mvcg, predictor, optimizer, loss_fn, max_feature_length, graphs)
        avg_epoch_loss = np.average(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_epoch_loss}')

        if((epoch + 1) % 1 == 0):
            # 测试
            print(f"Test {epoch + 1}")
            avg_recall, avg_mrr, avg_ndcg = test(test_dataloader, mvcg, predictor, graphs, max_feature_length, topk)
            if(avg_recall > maxRecall):
                maxRecall = avg_recall
            if(avg_mrr > maxMRR):
                maxMRR = avg_mrr
            if(avg_ndcg > maxNDCG):
                maxNDCG = avg_ndcg
            print(f'Epoch [{epoch + 1}/{num_epochs}], Recall: {avg_recall}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], MRR: {avg_mrr}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], NDCG: {avg_ndcg}')
            print(f"----------------------------------------------------------\n")

        # 调整学习率
        scheduler.step(avg_epoch_loss)

        # 早停
        early_stopping(avg_epoch_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(f'Max Recall: {maxRecall}')
    print(f'Max MRR: {maxMRR}')
    print(f'Max NDCG: {maxNDCG}')


if __name__ == '__main__':
    main()










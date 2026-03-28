import os
import sys
import dgl
import torch
import time
import datetime
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.hw.model_hw import MVCG, EarlyStopping, MyDataset, Predictor, TeeOutput


# parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_test_data_path = "../../dataset/processed/hw/processed_data_1.0.pth"
graph_data_path = "../../dataset/processed/hw/graphs_1.0.bin"
longtail_data_path = "../../dataset/processed/hw/longtail_data_threshold_10.npy"
logdir = "../../log/hw/hyperparameter/embedding_dimension"
isSaveModel = False
# model parameters
embedding_dim = 16
combiner_layer_num = 2
gnn_layer_num = 2
# train & test parameters
test_epoch = 2
batch_size = 128
learning_rate = 0.001
num_epochs = 30
topk_list = [5, 10, 20, 30, 40, 50]
# early_stopping parameters
es_patience = 20
es_delta = 0
es_verbose = False
# scheduler parameters
s_patience = 10
s_verbose = True
s_factor = 0.1
s_mode = 'min'


def train(dataloader, model, predictor, optimizer, loss_fn, max_feature_length, graphs):
    model.train()
    # predictor.train()
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


def mrr(pred, truth):
    reciprocal_rank = 0
    for rank, item in enumerate(pred, start=1):
        if item in truth:
            reciprocal_rank = 1 / rank
            break
    return reciprocal_rank


def ltr(pred, longtail_data, truth):
    count = 0
    longtail_data_set = set(longtail_data)
    truth_data_set = set(truth)
    correct_longtail_data_set = truth_data_set & longtail_data_set
    for i in pred:
        if i in correct_longtail_data_set:
            count += 1
    result = count / len(pred)
    return result


def test(dataloader, model, predictor, graphs, max_feature_length, longtail_data, topk_list):
    model.eval()
    with torch.no_grad():
        recall_values = {k: [] for k in topk_list}
        ndcg_values = {k: [] for k in topk_list}
        mrr_values = {k: [] for k in topk_list}
        ltr_values = {k: [] for k in topk_list}

        # 1. 预先计算所有item的embedding
        item_embeddings = []
        for i in range(len(dataloader)):
            embedding = model(graphs, torch.tensor(i).to(device), max_feature_length)
            item_embeddings.append(embedding[0])
        item_embeddings = torch.stack(item_embeddings, dim=0)  # shape: [num_items, embedding_dim]

        # 2. 测试
        for query, candidates, truths in tqdm(dataloader, desc="Testing"):
            query_id = query[0].item()
            query_embedding = item_embeddings[query_id]
            truths_tensor = torch.tensor(truths, dtype=torch.int32).to(device)  # 真标签
            truths_list = truths_tensor.tolist()
            candidates_ids = torch.tensor(candidates, dtype=torch.long).to(device)
            candidates_embeddings = item_embeddings[candidates_ids]

            scores = predictor(query_embedding.repeat(candidates_embeddings.size(0), 1), candidates_embeddings)

            for topk in topk_list:
                topk_scores, topk_indices = torch.topk(scores, k=topk, largest=True)
                topk_candidates = candidates_ids[topk_indices].tolist()

                recall_values[topk].append(recall(topk_candidates, truths_list))
                ndcg_values[topk].append(ndcg(topk_candidates, truths_list))
                mrr_values[topk].append(mrr(topk_candidates, truths_list))
                ltr_values[topk].append(ltr(topk_candidates, longtail_data, truths_list))

        avg_recalls = {k: sum(recall_values[k]) / len(recall_values[k]) for k in topk_list}
        avg_ndcgs = {k: sum(ndcg_values[k]) / len(ndcg_values[k]) for k in topk_list}
        avg_mrrs = {k: sum(mrr_values[k]) / len(mrr_values[k]) for k in topk_list}
        avg_ltrs = {k: sum(ltr_values[k]) / len(ltr_values[k]) for k in topk_list}

    return avg_recalls, avg_ndcgs, avg_mrrs, avg_ltrs


def log_to_file_with_terminal_output(log_dir=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 获取当前时间并生成文件名
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"log_{current_time}_edim{embedding_dim}_gcnlayer{gnn_layer_num}_alayer{combiner_layer_num}.txt"

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


# @log_to_file_with_terminal_output(log_dir=logdir)
def main():
    seed = 123  # 可以是任何数字，保持一致即可
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 如果使用GPU
    torch.backends.cudnn.deterministic = True


    # 读入数据
    loaded_data = torch.load(train_test_data_path)
    train_data = loaded_data["train_data"]
    test_data = loaded_data["test_data"]
    train_dataset = MyDataset(train_data)
    test_dataset = MyDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    strategies = loaded_data["strategies"]

    max_feature_length = loaded_data["max_feature_length"]

    graphs, _ = dgl.load_graphs(graph_data_path)
    graphs = [g.to(device) for g in graphs]

    longtail_data = np.load(longtail_data_path)

    node_nums = []
    for g in graphs:
        node_nums.append(g.number_of_nodes())
    print("Load data finish!")

    # 初始化模型
    predictor = Predictor(embedding_dim).to(device)  # 把predictor提取出来，为了加速测试阶段
    mvcg = MVCG(node_nums, strategies, device, embedding_dim, combiner_layer_num, gnn_layer_num).to(device)
    optimizer = torch.optim.Adam(list(mvcg.parameters()) + list(predictor.parameters()), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode=s_mode, factor=s_factor, patience=s_patience, verbose=s_verbose)
    early_stopping = EarlyStopping(es_patience, es_delta, es_verbose, path='./saved_model/early_stopping_checkpoint.pt')
    loss_fn = nn.BCEWithLogitsLoss().to(device)  # 用于二分类的交叉熵损失
    print("Initialize finish!")

    # # 加载模型
    # mvcg.load_state_dict(torch.load("./saved_model/best_ndcg@10_model.pth"))
    # predictor.load_state_dict(torch.load("./saved_model/best_ndcg@10_predictor.pth"))

    # 训练&测试模型
    training_efficiency_list = []
    training_loss_list = []
    maxRecall = {k: 0.0 for k in topk_list}
    maxNDCG = {k: 0.0 for k in topk_list}
    maxMRR = {k: 0.0 for k in topk_list}
    maxLTR = {k: 0.0 for k in topk_list}
    print(f'embedding_dim = {embedding_dim}')
    print(f'combiner_layer_num = {combiner_layer_num}')
    print(f'batch_size = {batch_size}')
    print(f'learning_rate = {learning_rate}')
    print(f'Running on {device}\n')
    start_time = time.perf_counter()
    for epoch in range(num_epochs):
        # 训练
        epoch_loss = train(train_dataloader, mvcg, predictor, optimizer, loss_fn, max_feature_length, graphs)
        avg_epoch_loss = np.average(epoch_loss)
        end_time = time.perf_counter()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_epoch_loss}, Train Time: {end_time - start_time}')
        training_efficiency_list.append([epoch + 1, end_time - start_time])
        training_loss_list.append([epoch + 1, avg_epoch_loss])

        # 测试
        if((epoch + 1) % test_epoch == 0):
            print(f"Test {epoch + 1}")
            avg_recalls, avg_ndcgs, avg_mrrs, avg_ltrs = test(test_dataloader, mvcg, predictor, graphs, max_feature_length, longtail_data, topk_list)
            for topk in topk_list:
                if avg_recalls[topk] > maxRecall[topk]:
                    maxRecall[topk] = avg_recalls[topk]
                    if topk == 10 and isSaveModel:
                        torch.save(mvcg.state_dict(), f'./saved_model/best_recall@10_model.pth')
                        torch.save(predictor.state_dict(), f'./saved_model/best_recall@10_predictor.pth')
                if avg_ndcgs[topk] > maxNDCG[topk]:
                    maxNDCG[topk] = avg_ndcgs[topk]
                    if topk == 10 and isSaveModel:
                        torch.save(mvcg.state_dict(), f'./saved_model/best_ndcg@10_model.pth')
                        torch.save(predictor.state_dict(), f'./saved_model/best_ndcg@10_predictor.pth')
                if avg_mrrs[topk] > maxMRR[topk]:
                    maxMRR[topk] = avg_mrrs[topk]
                    if topk == 10 and isSaveModel:
                        torch.save(mvcg.state_dict(), f'./saved_model/best_mrr@10_model.pth')
                        torch.save(predictor.state_dict(), f'./saved_model/best_mrr@10_predictor.pth')
                if avg_ltrs[topk] > maxLTR[topk]:
                    maxLTR[topk] = avg_ltrs[topk]
                print(f'Epoch [{epoch + 1}/{num_epochs}], Recall@{topk}: {avg_recalls[topk]}')
                print(f'Epoch [{epoch + 1}/{num_epochs}], NDCG@{topk}: {avg_ndcgs[topk]}')
                print(f'Epoch [{epoch + 1}/{num_epochs}], MRR@{topk}: {avg_mrrs[topk]}')
                print(f'Epoch [{epoch + 1}/{num_epochs}], LTR@{topk}: {avg_ltrs[topk]}')
            print(f"----------------------------------------------------------\n")


        # 调整学习率
        scheduler.step(avg_epoch_loss)

        # 早停
        early_stopping(avg_epoch_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # efficiency_df = pd.DataFrame(training_efficiency_list)
    # loss_df = pd.DataFrame(training_loss_list)
    # efficiency_df.to_excel(
    #     f'./train_analysis/graph_num/2/training_efficiency_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
    #     index=False)
    # loss_df.to_excel(
    #     f'./train_analysis/graph_num/2/training_loss_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
    #     index=False)

    print(f'Max Recall: {maxRecall}')
    print(f'Max NDCG: {maxNDCG}')
    print(f'Max MRR: {maxMRR}')
    print(f"Max LTR: {maxLTR}")


if __name__ == '__main__':
    main()










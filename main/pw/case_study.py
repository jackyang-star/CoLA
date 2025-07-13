import datetime
import os
import sys
import numpy as np
import random
import dgl
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from model.pw.model_pw import MVCG, MyDataset, Predictor, TeeOutput, SelfAttention


# parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = "../../dataset/processed/pw/processed_data_1.0.pth"
logdir = "../../log/pw/case_study"
# model parameters
embedding_dim = 64
combiner_layer_num = 2
gnn_layer_num = 2
# train & test parameters
test_epoch = 5
batch_size = 128
learning_rate = 0.001
num_epochs = 100
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


def test(dataloader, model, predictor, graphs, max_feature_length, topk_list):
    model.eval()
    attention_weights_storage = []
    attention_score_storage = []

    with torch.no_grad():
        recall_values = {k: [] for k in topk_list}
        mrr_values = {k: [] for k in topk_list}
        ndcg_values = {k: [] for k in topk_list}

        # 1. 预先计算所有item的embedding
        item_embeddings = []
        for i in range(len(dataloader)):
            index = i
            if index in {258, 801}:
                attention_weights_storage.clear()
                attention_score_storage.clear()
                temp_handles = []

                def save_attention_scores(module, input, output):
                    attention_weights_storage.append(module.latest_attention_weights.detach().cpu())
                    attention_score_storage.append(module.latest_attention_scores.detach().cpu())

                # 在index==258的时候，注册hook
                for name, module in model.named_modules():
                    if isinstance(module, SelfAttention):
                        handle = module.register_forward_hook(save_attention_scores)
                        temp_handles.append(handle)

                embedding = model(graphs, torch.tensor(index).to(device), max_feature_length)

                # 用完立刻remove hook
                for handle in temp_handles:
                    handle.remove()

                # print(f"Attention score for index {index} saved, number of layers: {len(attention_weights_storage)}")
                # for idx, att in enumerate(attention_weights_storage):
                #     print(f"Layer {idx}: attention weights matrix:")
                #     print(att)

                print(f"Attention score for index {index} saved, number of layers: {len(attention_weights_storage)}")
                for idx, att in enumerate(attention_weights_storage):
                    print(f"Layer {idx}: attention weights matrix:")
                    print(att)
            else:
                # 平时正常forward
                embedding = model(graphs, torch.tensor(index).to(device), max_feature_length)
            item_embeddings.append(embedding[0])
        item_embeddings = torch.stack(item_embeddings, dim=0)  # shape: [num_items, embedding_dim]

        # 2. 测试
        for query, candidates, truths in tqdm(dataloader, desc="Testing"):
            query_id = query[0].item()
            truths_tensor = torch.tensor(truths, dtype=torch.int32).to(device)  # 真标签
            truths_list = truths_tensor.tolist()

            query_embedding = item_embeddings[query_id]
            candidates_ids = torch.tensor(candidates, dtype=torch.long).to(device)
            candidates_embeddings = item_embeddings[candidates_ids]

            scores = predictor(query_embedding.repeat(candidates_embeddings.size(0), 1), candidates_embeddings)

            if query_id == 258 or query_id == 801:
                for topk in topk_list:
                    topk_scores, topk_indices = torch.topk(scores, k=topk, largest=True)
                    topk_candidates = candidates_ids[topk_indices].tolist()
                    correct = set(topk_candidates).intersection(set(truths_list))
                    print(f"[Query {query_id}] Top-{topk} Correct Predictions: {correct}")

                    recall_values[topk].append(recall(topk_candidates, truths_list))
                    mrr_values[topk].append(mrr(topk_candidates, truths_list))
                    ndcg_values[topk].append(ndcg(topk_candidates, truths_list))
            else:
                for topk in topk_list:
                    topk_scores, topk_indices = torch.topk(scores, k=topk, largest=True)
                    topk_candidates = candidates_ids[topk_indices].tolist()

                    recall_values[topk].append(recall(topk_candidates, truths_list))
                    mrr_values[topk].append(mrr(topk_candidates, truths_list))
                    ndcg_values[topk].append(ndcg(topk_candidates, truths_list))

        avg_recalls = {k: sum(recall_values[k]) / len(recall_values[k]) for k in topk_list}
        avg_mrrs = {k: sum(mrr_values[k]) / len(mrr_values[k]) for k in topk_list}
        avg_ndcgs = {k: sum(ndcg_values[k]) / len(ndcg_values[k]) for k in topk_list}

    return avg_recalls, avg_mrrs, avg_ndcgs


def log_to_file_with_terminal_output(log_dir=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 获取当前时间并生成文件名
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"log_{current_time}_edim{embedding_dim}_alayer{combiner_layer_num}_batchsize{batch_size}.txt"

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


@log_to_file_with_terminal_output(log_dir=logdir)
def main():
    seed = 123  # 可以是任何数字，保持一致即可
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 如果使用GPU
    torch.backends.cudnn.deterministic = True

    # 读入数据
    loaded_data = torch.load(data_path)
    test_data = loaded_data["test_data"]
    strategies = loaded_data["strategies"]
    max_feature_length = loaded_data["max_feature_length"]
    graphs, _ = dgl.load_graphs("../../dataset/processed/pw/graphs_1.0.bin")
    graphs = [g.to(device) for g in graphs]
    test_dataset = MyDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # 初始化模型
    node_nums = []
    for g in graphs:
        node_nums.append(g.number_of_nodes())
    predictor = Predictor(embedding_dim).to(device)  # 把predictor提取出来，为了加速测试阶段
    mvcg = MVCG(node_nums, strategies, device, embedding_dim, combiner_layer_num, gnn_layer_num).to(device)
    mvcg.load_state_dict(torch.load('./saved_model/best_recall@10_model.pth'))
    mvcg.eval()
    predictor.load_state_dict(torch.load('./saved_model/best_recall@10_predictor.pth'))
    predictor.eval()

    # 测试模型
    maxRecall = {k: 0.0 for k in topk_list}
    maxMRR = {k: 0.0 for k in topk_list}
    maxNDCG = {k: 0.0 for k in topk_list}

    print(f'embedding_dim = {embedding_dim}')
    print(f'combiner_layer_num = {combiner_layer_num}')
    print(f'batch_size = {batch_size}')
    print(f'learning_rate = {learning_rate}')
    print(f'Running on {device}\n')


    avg_recalls, avg_mrrs, avg_ndcgs = test(test_dataloader, mvcg, predictor, graphs, max_feature_length, topk_list)
    for topk in topk_list:
        if avg_recalls[topk] > maxRecall[topk]:
            maxRecall[topk] = avg_recalls[topk]
        if avg_mrrs[topk] > maxMRR[topk]:
            maxMRR[topk] = avg_mrrs[topk]
        if avg_ndcgs[topk] > maxNDCG[topk]:
            maxNDCG[topk] = avg_ndcgs[topk]
        print(f'Recall@{topk}: {avg_recalls[topk]}')
        print(f'MRR@{topk}: {avg_mrrs[topk]}')
        print(f'NDCG@{topk}: {avg_ndcgs[topk]}')
    print(f"----------------------------------------------------------\n")


    print(f'Max Recall: {maxRecall}')
    print(f'Max MRR: {maxMRR}')
    print(f'Max NDCG: {maxNDCG}')


if __name__ == '__main__':
    main()










import torch
import numpy as np
from model.model import MVCG


# 参数
emb_dim = 64
topk = 10
train_test_ratio = 0.7
dropout_rate = 0.1
head_num = 8
iftrain = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 精确度指标
# Recall指标
def recall(y_true, y_pred):
    recalls = []
    for true_items, pred_items in zip(y_true, y_pred):
        true_positives = len(set(true_items) & set(pred_items))
        recalls.append(true_positives / len(true_items))
    return np.mean(recalls)

# 排序指标
# DCG计算
def dcg_at_k(r):
    r = np.asfarray(r)
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0
# NDCG指标
def ndcg_at_k(y_true, y_pred):
    ndcgs = []
    for true_items, pred_items in zip(y_true, y_pred):
        ideal = [1] * len(set(true_items) & set(pred_items))
        actual = [1 if item in true_items else 0 for item in pred_items]
        idcg = dcg_at_k(ideal)
        dcg = dcg_at_k(actual)
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    return np.mean(ndcgs)

# MRR指标
def mrr(y_true, y_pred):
    mrrs = []
    for true_items, pred_items in zip(y_true, y_pred):
        reciprocal_rank = 0
        for rank, item in enumerate(pred_items, start=1):
            if item in true_items:
                reciprocal_rank = 1 / rank
                break
        mrrs.append(reciprocal_rank)
    return np.mean(mrrs)


def execute_test():
    # 加载测试数据集
    api_emb_data = np.load('../dataset/processed/dataset/api_emb_dataset.npy', allow_pickle=True)
    gt_data = np.load('../dataset/processed/dataset/gt_dataset.npy', allow_pickle=True)

    # 初始化并加载模型
    model = MVCG(emb_dim, dropout_rate, head_num, iftrain).to(device)
    model.load_state_dict(torch.load('../model/checkpoint.pt'))

    # 测试
    candidates_comp = np.zeros((len(gt_data), len(gt_data)))
    prediction = []
    truth = []

    print(f'Test begin. Running on {device}')
    model.eval()
    with torch.no_grad():
        row_idx = 0
        for query_idx, query_emb in api_emb_data:
            query_emb = torch.tensor(query_emb, dtype=torch.float32).to(device)
            col_idx = 0
            for candidate_idx, candidate_emb in api_emb_data:
                candidate_emb = torch.tensor(candidate_emb, dtype=torch.float32).to(device)
                # 计算并保存相似度
                candidates_comp[row_idx, col_idx] = model(query_emb, candidate_emb)
                col_idx += 1

            # 获取真实互补编号(剔除正样本后的)
            gt = gt_data[gt_data[:, 0] == query_idx, 1][0]
            pos_len = int(len(gt) * train_test_ratio)
            gt = gt[pos_len:]
            deleted = gt[:pos_len]
            truth.append(gt)

            # 获取预测TopK的编号(剔除正样本后的)
            topk_indices = []
            sorted_indices = np.argsort(candidates_comp[row_idx])[::-1]
            count = 1
            for index in sorted_indices:
                if count > topk:
                    break
                if index in deleted:
                    pass
                else:
                    topk_indices.append(index)
                    count += 1
            prediction.append(topk_indices)
            row_idx += 1

    print(f"Recall@{topk}: %.6f" %recall(truth, prediction))
    print(f"NDCG@{topk}: %.6f" %ndcg_at_k(truth, prediction))
    print(f"MRR@{topk}: %.6f" %mrr(truth, prediction))


if __name__ == '__main__':
    execute_test()










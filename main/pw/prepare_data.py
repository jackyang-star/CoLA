import os
import random
import torch
import dgl
import pandas as pd
import numpy as np
import scipy.sparse as sp

from utils.create_cm_fv_utils import create_fv_list, create_api_cooccurrence_matrix, create_fv_cooccurrence_matrix


name = 'Name'
relatedapi = 'Related APIs'
train_test_ratio = 0.8
api_path = '../../dataset/raw/programmableweb/apiData.json'
mashup_path = '../../dataset/raw/programmableweb/mashupData.json'
feature_dir = '../../dataset/processed/pw/comp_feature/'
save_dir = '../../dataset/processed/pw/'
features_files = {
    '0': '0_features.npy',
    '0.2': '0.2_features.npy',
    '0.4': '0.4_features.npy',
    '0.6': '0.6_features.npy',
    '0.8': '0.8_features.npy',
    '1.0': '1.0_features.npy',
    '1.2': '1.2_features.npy',
    '1.4': '1.4_features.npy',
}


def create_api_cm(api_df, mashup_df):
    api_list = sorted(api_df[name].tolist())
    api_cm_array = create_api_cooccurrence_matrix(mashup_df, api_list, relatedapi)

    # 只保留非零行和非零列
    non_zero_rows = np.any(api_cm_array != 0, axis=1)
    non_zero_cols = np.any(api_cm_array != 0, axis=0)
    non_zero_row_indices = np.nonzero(non_zero_rows)[0]
    api_list_array = np.array([api_list[i] for i in non_zero_row_indices])
    api_cm_array = api_cm_array[non_zero_rows][:, non_zero_cols]

    return api_list_array, api_cm_array

def create_dgl_weighted_graph(adj_matrix):
    adj_matrix = adj_matrix.astype(np.int32)
    sp_matrix = sp.coo_matrix(adj_matrix)
    g = dgl.from_scipy(sp_matrix, eweight_name='weight')
    g.edata['weight'] = g.edata['weight'].float()
    return g

def create_graphs(features, api_df, api_list_array, api_cm_array, test_data):
    # 删除测试数据
    # for data in test_data:
    #     for truth in data[2]:
    #         api_cm_array[data[0][0], truth] = 0

    max_feature_length = 0
    graphs = []
    for feature in features:
        fv_list_array = create_fv_list(api_df, api_list_array, feature, name)
        cm_array, max_fv_len = create_fv_cooccurrence_matrix(api_cm_array, api_list_array, api_df, fv_list_array, feature, name)
        # 删除测试数据
        if feature == name:
            for data in test_data:
                for truth in data[2]:
                    cm_array[data[0][0], truth] = 0
        graphs.append(create_dgl_weighted_graph(cm_array))
        max_feature_length = max(max_feature_length, max_fv_len)
    return graphs, max_feature_length

def prepare_data(api_cm_array, api_list_array, api_df, features, advanced=False):
    train_data, test_data, strategies = [], [], []
    fv_dict = {feature: create_fv_list(api_df, api_list_array, feature, name) for feature in features}
    test_set = set()

    for i in range(api_cm_array.shape[0]):
        non_zero_columns = np.nonzero(api_cm_array[i])[0].tolist()
        zero_columns = np.where(api_cm_array[i] == 0)[0].tolist()
        train_length = int(len(non_zero_columns) * train_test_ratio)

        # 准备测试数据
        truths = non_zero_columns[train_length:]
        all_columns = list(range(api_cm_array.shape[1]))
        candidates = list(set(all_columns) - set(non_zero_columns[:train_length]) - {i})
        test_data.append(([i], candidates, truths))

        # 准备训练数据
        for truth in truths:
            test_set.add((i, truth))

        positives = non_zero_columns[:train_length]
        negatives = [col for col in zero_columns if col != i and (i, col) not in test_set and (col, i) not in test_set]
        negatives = random.sample(negatives, min(train_length, len(negatives)))

        for pos in positives:
            if not advanced or ((i, pos) not in test_set and (pos, i) not in test_set):
                train_data.append((i, pos, 1))
        for neg in negatives:
            train_data.append((i, neg, 0))

        # 保存云API与特征值的对应关系
        api_name = api_list_array[i]
        matching_api = api_df[api_df[name] == api_name]
        item = {}
        for idx, feature in enumerate(features):
            matching_feature = matching_api[feature].iloc[0]
            if not pd.isna(matching_feature):
                indexes = []
                for fv in map(str.strip, matching_feature.split(',')):
                    fv_index = np.where(fv_dict[feature] == fv)[0][0]
                    indexes.append(fv_index)
                item[idx] = indexes
        strategies.append(item)

    return train_data, test_data, strategies


def prepare_longtail(api_list, mashup_df, longtail_threshold):
    pw_api_map = {api: 0 for api in api_list}
    for items in mashup_df[relatedapi]:
        items = items.split(',')
        for item in items:
            item = item.strip()
            if item in pw_api_map:
                pw_api_map[item] += 1

    pw_api_map = {k: v for k, v in pw_api_map.items() if v > 0}

    longtail_indices = [
        idx for idx, api in enumerate(api_list)
        if pw_api_map.get(api, 0) < longtail_threshold and pw_api_map.get(api, 0) > 0
    ]
    print(longtail_indices)
    print(f"[{len(longtail_indices)}/{len(api_list)}]")

    return longtail_indices


def main(features_key, longtail_threshold=4, ablation=None, advanced=False):
    # 控制随机性
    random.seed(123)
    # 读数据
    api_df = pd.read_json(api_path)
    mashup_df = pd.read_json(mashup_path)
    features = np.load(os.path.join(feature_dir, features_files[features_key])).tolist()
    if ablation == "no_name":
        features = [x for x in features if x != "Name"]
    if ablation == "no_pc":
        features = [x for x in features if x != "Primary Category"]
    if ablation == "no_sc":
        features = [x for x in features if x != "Secondary Categories"]
    if ablation == "no_pr":
        features = [x for x in features if x != "API Provider"]

    # 生成API列表和共现矩阵
    api_list_array, api_cm_array = create_api_cm(api_df, mashup_df)

    # 准备数据
    train_data, test_data, strategies = prepare_data(api_cm_array, api_list_array, api_df, features, advanced)
    graphs, max_feature_length = create_graphs(features, api_df, api_list_array, api_cm_array, test_data)
    longtail_data = prepare_longtail(api_list_array, mashup_df, longtail_threshold)

    # 保存处理后的数据
    # if ablation != None:
    #     torch.save({
    #         "train_data": train_data,
    #         "test_data": test_data,
    #         "strategies": strategies,
    #         "max_feature_length": max_feature_length,
    #     }, os.path.join(save_dir, f"processed_data_{features_key}_{ablation}.pth"))
    #     dgl.save_graphs(os.path.join(save_dir, f"graphs_{features_key}_{ablation}.bin"), graphs)
    #     print(f"Saved processed data and graphs for feature {features_key}_{ablation}")
    # else:
    #     torch.save({
    #         "train_data": train_data,
    #         "test_data": test_data,
    #         "strategies": strategies,
    #         "max_feature_length": max_feature_length,
    #     }, os.path.join(save_dir, f"processed_data_{features_key}.pth"))
    #     dgl.save_graphs(os.path.join(save_dir, f"graphs_{features_key}.bin"), graphs)
    #     print(f"Saved processed data and graphs for feature {features_key}")
    np.save(os.path.join(save_dir, f"longtail_data_threshold_{longtail_threshold}.npy"), longtail_data)

if __name__ == "__main__":
    main('1.0', 2)

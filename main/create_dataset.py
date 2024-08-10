import json
import numpy as np
import pandas as pd


# 参数
train_test_ratio = 0.7
emb_dim = 64


api_cm_array = np.load('../dataset/processed/cm_fv/api_cm.npy')  # 读入API共现矩阵和列表
api_list_array = np.load('../dataset/processed/cm_fv/api_list.npy')  # 读入API列表
api_df = pd.read_json('../dataset/raw/programmableweb/apiData.json')  # 读入API数据DataFrame
features_list = np.load('../dataset/processed/comp_feature/entropy_c_features.npy').tolist()  # 读入互补特征列表
# noname_features = np.load('../dataset/processed/comp_feature/noname_entropy_c_features.npy').tolist()
# noother_features = np.load('../dataset/processed/comp_feature/noother_entropy_c_features.npy').tolist()
# 读入互补特征的特征值列表
fv_list_dict = {}
for feature in features_list:
    fv_list_array = np.load(f'../dataset/processed/cm_fv/{feature}_list.npy').tolist()
    fv_list_dict[feature] = fv_list_array  # 将特征列表添加到字典中
# 读入各个特征的特征值的embedding
with open("../dataset/processed/embeddings/gcn_embeddings.json", "r") as json_file:
    loaded_embeddings_dict = json.load(json_file)
# 读入最大特征值个数
max_fv_len = np.loadtxt('../dataset/processed/cm_fv/max_fv_length.txt').astype(int)
print('Loading File Finish')


# 创建Y
gt_list = []
for i in range(api_cm_array.shape[0]):
    non_zero_columns = np.nonzero(api_cm_array[i])[0].tolist()
    gt_list.append([i, non_zero_columns])
print('Creating gt_list Finish')


# 构造API编号与embedding的列表
fv_list = []  # API编号与embedding的列表
for i in range(api_cm_array.shape[0]):
    feature_emb = []
    api_name = api_list_array[i]  # 根据行号，从api_list中找到api的名称
    matching_api = api_df[api_df['Name'] == api_name]  # 根据api_name，从api_df中找到对应的行
    for feature in features_list:
        fv_emb_array = np.zeros((max_fv_len, emb_dim), dtype=np.float32)  # 用于保存某一特征所有特征值的embedding的列表
        index = 0
        matching_feature = matching_api[feature].iloc[0]  # 获取特征值
        if not pd.isna(matching_feature):
            matching_feature = [e.strip() for e in matching_feature.split(',')]
            for fv in matching_feature:
                fv_index = fv_list_dict[feature].index(fv)  # 通过特征值的名称在特征值列表中找到特征值的编号
                fv_emb = loaded_embeddings_dict[feature][fv_index]  # 通过编号找到embedding
                fv_emb_array[index, :] = fv_emb
                index += 1
        feature_emb.append(fv_emb_array.tolist())
    fv_list.append([i, feature_emb])
print('Creating fv_list Finish')


# 构造训练数据
train_list = []  # API的embedding与训练embedding的列表
for row, pos_cols in gt_list:
    row_emb = [item[1] for item in fv_list if item[0] == row][0]  # 查找行embedding
    # 随机负采样
    train_len = int(len(pos_cols) * train_test_ratio)
    neg_cols = np.where(api_cm_array[row] == 0)[0]
    neg_length = len(neg_cols)
    neg_cols = np.random.choice(neg_cols, int(neg_length * 0.5), replace=False)
    # 构造正样本训练数据
    for pos in pos_cols[:train_len]:
        pos_emb = [item[1] for item in fv_list if item[0] == pos][0]
        train_list.append([row_emb, pos_emb, 1.0])
    # 构造负样本训练数据
    for neg in neg_cols:
        neg_emb = [item[1] for item in fv_list if item[0] == neg][0]
        train_list.append([row_emb, neg_emb, 0.0])
print('Creating train_list Finish')


# 保存为npy文件
gt_array = np.array(gt_list, dtype=object)
fv_array = np.array(fv_list, dtype=object)
train_array = np.array(train_list, dtype=object)
np.save('../dataset/processed/dataset/gt_dataset.npy', gt_array)
np.save('../dataset/processed/dataset/api_emb_dataset.npy', fv_array)
np.save('../dataset/processed/dataset/train_dataset.npy', train_array)

# 保存为csv文件
gt_df = pd.DataFrame(gt_list)
fv_df = pd.DataFrame(fv_list)
train_df = pd.DataFrame(train_list)
gt_df.to_csv('../dataset/processed/dataset/gt_dataset.csv', index=False)
fv_df.to_csv('../dataset/processed/dataset/api_emb_dataset.csv', index=False)
train_df.to_csv('../dataset/processed/dataset/train_dataset.csv', index=False)










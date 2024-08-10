import pandas as pd
import numpy as np
from utils.create_cm_fv_utils import create_fv_list, create_api_cooccurrence_matrix, create_fv_cooccurrence_matrix


# 读入文件
api_df = pd.read_json('../dataset/raw/programmableweb/apiData.json')
mashup_df = pd.read_json('../dataset/raw/programmableweb/mashupData.json')
features = np.load('../dataset/processed/comp_feature/entropy_c_features.npy').tolist()
noname_features = np.load('../dataset/processed/comp_feature/noname_entropy_c_features.npy').tolist()
noother_features = np.load('../dataset/processed/comp_feature/noother_entropy_c_features.npy').tolist()
print('Loading File Finish')


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
# 保存为npy文件
np.save('../dataset/processed/cm_fv/api_list.npy', api_list_array)
np.save('../dataset/processed/cm_fv/api_cm.npy', api_cm_array)
# 保存为csv文件
api_list_df = pd.DataFrame(api_list_array)
api_cm_df = pd.DataFrame(api_cm_array)
api_list_df.to_csv('../dataset/processed/cm_fv/api_list.csv', index=False)
api_cm_df.to_csv('../dataset/processed/cm_fv/api_cm.csv', index=False)

# 查看API基本情况
print(f'api_list Number: {len(api_list_array)}')
zero_rows = []
for i in range(api_cm_array.shape[0]):
    if np.all(api_cm_array[i] == 0):
        zero_rows.append(i)
print('API CM')
print(f'最大值：{np.max(api_cm_array)}')
print(f'总共的结点数：{api_cm_array.shape[0]}')
print("不存在边的结点的个数：", len(zero_rows))
print('-----------------------------------------------------------------------------------------------------------')


# 为每个feature_value保存列表、特征值最大数量和共现矩阵
max_length = 0
for feature in features:
    # 设置保存的文件的名称
    list_file_name = f"{feature}_list"
    cm_file_name = f"{feature}_cm"

    fv_list_array = create_fv_list(api_df, api_list_array, feature)  # 获取feature_value列表
    cm_array, max_fv_len = create_fv_cooccurrence_matrix(api_cm_array, api_list_array, api_df, fv_list_array, feature)  # 构建feature_value共现矩阵
    if max_fv_len > max_length:
        max_length = max_fv_len  # 获取特征值最大数量

    # 保存feature_value列表和feature_value共现矩阵
    # 保存为npy格式
    np.save(f'../dataset/processed/cm_fv/{list_file_name}.npy', fv_list_array)
    np.save(f'../dataset/processed/cm_fv/{cm_file_name}.npy', cm_array)
    #保存为csv格式
    fv_list_df = pd.DataFrame(fv_list_array)
    cm_array_df = pd.DataFrame(cm_array)
    fv_list_df.to_csv(f'../dataset/processed/cm_fv/{list_file_name}.csv', index=False)
    cm_array_df.to_csv(f'../dataset/processed/cm_fv/{cm_file_name}.csv', index=False)

    # 查看fv_list基本情况
    print(f"{feature} fv_list Number: {len(fv_list_array)}")
    # 查看共现矩阵基本情况
    zero_rows = []
    for i in range(cm_array.shape[0]):
        if np.all(cm_array[i] == 0):
            zero_rows.append(i)
    print(f'{feature} CM')
    print(f'最大值：{np.max(cm_array)}')
    print(f'总共的结点数：{cm_array.shape[0]}')
    print("不存在边的结点的个数：", len(zero_rows))
    print('-----------------------------------------------------------------------------------------------------------')
np.savetxt(f'../dataset/processed/cm_fv/max_fv_length.txt', np.array([max_length]), fmt='%d')  # 保存特征值最大数量










import numpy as np
from itertools import combinations


# feature_values列表的提取
def create_fv_list(api_df, api_list_array, feature):
    values_list = []
    for api_name in api_list_array:
        matching_api = api_df[api_df['Name'] == api_name]
        fv = matching_api[feature].iloc[0]
        if type(fv) == str:
            fv_list = [value.strip() for value in fv.split(',')]
            values_list.extend(fv_list)
    # for values in api_df[feature]:
    #     if type(values) == str:
    #         temp_list = [value.strip() for value in values.split(',')]  # 把字符串类型数据按list类型保存
    #         values_list.extend(temp_list)
    values_list = sorted(list(set(values_list)))  # 去重&按字典顺序排序
    values_array = np.array(values_list)
    return values_array


# 构建API共现矩阵
def create_api_cooccurrence_matrix(mashup_df, api_list, feature_name):
    # 构建空的共现矩阵
    cooccurrence_matrix = np.zeros((len(api_list), len(api_list)), dtype=np.int32)
    # 填充共现矩阵
    for values in mashup_df[feature_name]:  # 遍历每一个mashup的Related APIs
        temp_list = [value.strip() for value in values.split(',')]
        for value_name1, value_name2 in combinations(temp_list, 2):  # 两两组合temp_list中的API，填充共现矩阵
            if value_name1 in api_list and value_name2 in api_list:  # 不在api_list中的API无法填充共现矩阵
                x = api_list.index(value_name1)
                y = api_list.index(value_name2)
                cooccurrence_matrix[x, y] += 1
                cooccurrence_matrix[y, x] += 1  # 对称性
    return cooccurrence_matrix


# 由API的共现矩阵构建feature_value的共现矩阵
def create_fv_cooccurrence_matrix(api_cm_array, api_list_array, api_df, fv_list_array, feature):
    # 构建空的共现矩阵&初始化max_fv_len
    max_fv_len = 0
    cm_array = np.zeros((len(fv_list_array), len(fv_list_array)), dtype=np.int32)
    # 预处理数据，把API和对应的feature_value保存为字典形式
    api_name_to_feature_values = {}
    for i in range(len(api_list_array)):
        api_name = api_list_array[i]
        matching_api = api_df[api_df['Name'] == api_name]
        feature_values = matching_api[feature].unique()
        if len(feature_values) > 0:
            feature_values = [fv.strip() for fv in str(feature_values[0]).split(',')]
            api_name_to_feature_values[api_name] = set(feature_values)
        if len(feature_values) > max_fv_len:
            max_fv_len = len(feature_values)
    # 填充共现矩阵
    rows, cols = api_cm_array.shape
    for i in range(rows):
        for j in range(i, rows):  # j从i开始遍历，避免重复计算
            if api_cm_array[i, j] > 0:
                # 获取对应位置API的名称
                api1_name = api_list_array[i]
                api2_name = api_list_array[j]
                # 获取API的特征值
                api1_feature_values = api_name_to_feature_values.get(api1_name, set())
                api2_feature_values = api_name_to_feature_values.get(api2_name, set())
                # 若没有共同的特征值，则存在互补关系
                if not api1_feature_values & api2_feature_values:
                    # 记录特征值的共现关系
                    for fv1 in api1_feature_values:  # 特征值两两建立连接是否合适？？？
                        for fv2 in api2_feature_values:
                            # 判断特征值是否在特征值列表中(用于排除nan)
                            if fv1 in fv_list_array and fv2 in fv_list_array:
                                x = np.where(fv_list_array == fv1)[0][0]
                                y = np.where(fv_list_array == fv2)[0][0]
                                if x != y:
                                    cm_array[x, y] += 1
                                    cm_array[y, x] += 1
    return cm_array, max_fv_len










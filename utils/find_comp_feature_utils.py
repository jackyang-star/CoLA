from itertools import combinations
import pandas as pd
import numpy as np


# 定义计算Entropy的函数
def entropy(series):
    p = series.value_counts(normalize=True)
    return -(p * np.log2(p + 1e-10)).sum()

# 定义计算Gini的函数
# def gini(series):
#     if series.empty:
#         p = 1
#     else:
#         p = series.value_counts(normalize=True)
#     return 1 - np.sum(p ** 2)


def find(api_df, mashup_df, m_features, entropy_threshold, gini_threshold):
    ma_entropy = {}
    # ma_gini = {}
    feature_entropies = {}
    # feature_ginies = {}

    # 遍历mashup
    for _, mashup in mashup_df.iterrows():
        api_list = [api.strip() for api in str(mashup['Related APIs']).split(',')]  # 获取mashup所调用的API列表
        if len(api_list) >= 2:
            # 遍历候选特征,计算该mashup在该特征的多样性指标并记录
            for feature in m_features:
                fv_pair = [(0, 0)]  # 用于记录互补取值的对
                for api_name1, api_name2 in combinations(api_list, 2):
                    matching_api1 = api_df[api_df['Name'] == api_name1]
                    matching_api2 = api_df[api_df['Name'] == api_name2]
                    if not matching_api1.empty and not matching_api2.empty:  # 存在被mashup调用但是没有特征的API
                        if feature in matching_api1.columns and feature in matching_api2.columns:
                            # 处理多特征值
                            matching1_fv = [e.strip() for e in str(matching_api1[feature].iloc[0]).split(',')]
                            matching2_fv = [e.strip() for e in str(matching_api2[feature].iloc[0]).split(',')]
                            if matching1_fv and matching2_fv:
                                matching1_fv_set = set(matching1_fv)
                                matching2_fv_set = set(matching2_fv)
                                if not matching1_fv_set & matching2_fv_set:
                                    if (matching1_fv_set, matching2_fv_set) not in fv_pair:  # 防止重复添加
                                        fv_pair.append((matching1_fv_set, matching2_fv_set))
                feature_values_entropy = entropy(pd.Series(fv_pair))
                # feature_values_gini = gini(pd.Series(fv_pair))
                ma_entropy[(mashup['Name'], feature)] = feature_values_entropy
                # ma_gini[(mashup['Name'], feature)] = feature_values_gini

    # 取每个feature在所有mashup上的entropy，求平均值作为该feature的entropy
    for feature in m_features:
        # if feature in api_df.columns:
        feature_entropies[feature] = np.mean([ma_entropy[(mashup_id, feature)] for mashup_id, f in ma_entropy.keys() if f == feature])
        # feature_ginies[feature] = np.mean([ma_gini[(mashup_id, feature)] for mashup_id, f in ma_gini.keys() if f == feature])
        print(f'Feature: {feature}\nEntropy: {feature_entropies[feature]}')

    # 判断特征的互补性
    c_features = []
    for threshold in entropy_threshold:
        entropy_complementary_features = [feature for feature, entropy_value in feature_entropies.items() if
                                          entropy_value > threshold]
        c_features.append(entropy_complementary_features)
    # gini_complementary_features = [feature for feature, gini_value in feature_ginies.items() if
    #                                   gini_value > gini_threshold]

    return c_features









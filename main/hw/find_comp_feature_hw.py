import numpy as np
import pandas as pd

from utils.find_comp_feature_utils import find


# 参数
features = ['name', 'category']
entropy_threshold = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
relatedapi = 'relatedAPIs'
name = 'name'
api_path = '../../dataset/raw/HuaWei/api.json'
mashup_path = '../../dataset/raw/HuaWei/app.json'


# 读入文件
api_df = pd.read_json(api_path)
mashup_df = pd.read_json(mashup_path)
print('Loading File Finish')


# 求出互补特征
entropy_c_features, feature_entropies = find(api_df, mashup_df, features, entropy_threshold, relatedapi, name)
# 保存互补特征
for c_feature, threshold in zip(entropy_c_features, entropy_threshold):
    # 保存为csv文件
    e_df = pd.DataFrame(c_feature)
    e_df.to_csv(f'../../dataset/processed/hw/comp_feature/{threshold}_features.csv', index=False)
    # 保存为npy文件
    e_array = np.array(c_feature)
    np.save(f'../../dataset/processed/hw/comp_feature/{threshold}_features.npy', e_array)
# 保存互补特征的熵
with open("../../dataset/processed/hw/comp_feature/feature_entropies.txt", "w", encoding="utf-8") as file:
    for key, value in feature_entropies.items():
        file.write(f"Feature: {key}\nEntropy: {value}\n")
# 打印特征的熵和互补特征
print(feature_entropies)
print(f'entropy_c_features: {entropy_c_features}')










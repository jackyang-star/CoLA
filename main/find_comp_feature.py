import numpy as np
import pandas as pd
from utils.find_comp_feature_utils import find


# 参数
features = ['Name', 'Primary Category', 'Secondary Categories', 'API Provider', 'Authentication model', 'Version status',
            'Type', 'Scope', 'Architectural Style', 'Supported Request Formats', 'Supported Response Formats']
entropy_threshold = 1
gini_threshold = 0.4


# 读入文件
api_df = pd.read_json('../dataset/raw/programmableweb/apiData.json')
mashup_df = pd.read_json('../dataset/raw/programmableweb/mashupData.json')
print('Loading File Finish')


# 求出互补特征
entropy_c_features, gini_c_features = find(api_df, mashup_df, features, entropy_threshold, gini_threshold)
# 保存互补特征
# 保存为csv文件
e_df = pd.DataFrame(entropy_c_features)
g_df = pd.DataFrame(gini_c_features)
e_df.to_csv('../dataset/processed/comp_feature/entropy_c_features.csv', index=False)
g_df.to_csv('../dataset/processed/comp_feature/gini_c_features.csv', index=False)
# 保存为npy文件
e_array = np.array(entropy_c_features)
g_array = np.array(gini_c_features)
np.save('../dataset/processed/comp_feature/entropy_c_features.npy', e_array)
np.save('../dataset/processed/comp_feature/gini_c_features.npy', g_array)
# 打印互补特征
print(f'entropy_c_features: {entropy_c_features}')
print(f'gini_c_features: {gini_c_features}')










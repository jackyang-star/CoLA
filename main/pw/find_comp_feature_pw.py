import numpy as np
import pandas as pd
from utils.find_comp_feature_utils import find


# 参数
features = ['Name', 'Primary Category', 'Secondary Categories', 'API Provider', 'Version status', 'Type', 'Scope',
            'Architectural Style', 'Supported Request Formats', 'Supported Response Formats']
entropy_threshold = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
relatedapi = 'Related API'
name = 'Name'


# 读入文件
api_df = pd.read_json('../../dataset/raw/programmableweb/apiData.json')
mashup_df = pd.read_json('../../dataset/raw/programmableweb/mashupData.json')
print('Loading File Finish')


# 求出互补特征
entropy_c_features = find(api_df, mashup_df, features, entropy_threshold, relatedapi, name)
# 保存互补特征
for c_feature, threshold in zip(entropy_c_features, entropy_threshold):
    # 保存为csv文件
    e_df = pd.DataFrame(c_feature)
    e_df.to_csv(f'../dataset/processed/pw/comp_feature/{threshold}_features.csv', index=False)
    # 保存为npy文件
    e_array = np.array(c_feature)
    np.save(f'../dataset/processed/pw/comp_feature/{threshold}_features.npy', e_array)
    # 打印互补特征
    print(f'entropy_c_features: {entropy_c_features}')










import numpy as np
import pandas as pd

from utils.find_comp_feature_utils import find


# 参数
features = ['name', 'category']
entropy_threshold = [1.0]
relatedapi = 'relatedAPIs'
name = 'name'


# 读入文件
api_df = pd.read_json('../../dataset/raw/HuaWei/api.json')
mashup_df = pd.read_json('../../dataset/raw/HuaWei/app.json')
print('Loading File Finish')


# 求出互补特征
entropy_c_features = find(api_df, mashup_df, features, entropy_threshold, relatedapi, name)
# 保存互补特征
for c_feature in entropy_c_features:
    # 保存为csv文件
    e_df = pd.DataFrame(c_feature)
    e_df.to_csv(f'../dataset/processed/hw/comp_feature/comp_features.csv', index=False)
    # 保存为npy文件
    e_array = np.array(c_feature)
    np.save(f'../../dataset/processed/hw/comp_feature/comp_features.npy', e_array)
    # 打印互补特征
    print(f'entropy_c_features: {c_feature}')










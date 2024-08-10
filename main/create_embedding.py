import json
import numpy as np
from model.model import GCN


# 参数
emb_dim = 64
layer_num = 2


# 读入互补特征
features = np.load('../dataset/processed/comp_feature/entropy_c_features.npy').tolist()
# noname_features = np.load('../dataset/processed/comp_feature/noname_entropy_c_features.npy').tolist()
# noother_features = np.load('../dataset/processed/comp_feature/noother_entropy_c_features.npy').tolist()
print('File Read Finish')


embedding_dict = {}
for feature in features:
    cm_array = np.load(f'../dataset/processed/cm_fv/{feature}_cm.npy')
    n_nodes = cm_array.shape[0]
    gcn = GCN(n_nodes, emb_dim, layer_num)
    embeddings = gcn(cm_array)
    embedding_dict[feature] = embeddings.tolist()
print('Creating Embeddings Finish')

# 将字典数据保存到JSON文件中
with open("../dataset/processed/embeddings/gcn_embeddings.json", "w") as json_file:
    json.dump(embedding_dict, json_file)
print('Saving File Finish')










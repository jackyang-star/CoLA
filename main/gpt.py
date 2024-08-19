import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import SAGEConv
from torch.utils.data import DataLoader, Dataset

# GraphSAGE模型，生成物品嵌入
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, h_feats, aggregator_type='mean')

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

# 自注意力机制，用于组合不同图的嵌入
class SelfAttentionCombiner(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttentionCombiner, self).__init__()
        self.attn_weights = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        nn.init.xavier_uniform_(self.attn_weights.data)

    def forward(self, embeddings):
        attn_scores = torch.einsum('cd,de->ce', embeddings, self.attn_weights)
        attn_scores = torch.mean(attn_scores, dim=-1)
        attn_scores = F.softmax(attn_scores, dim=0)
        combined_embedding = torch.einsum('cd,c->d', embeddings, attn_scores)
        return combined_embedding

# 端到端模型
class EndToEndModel(nn.Module):
    def __init__(self, gnn_models, attn_combiners):
        super(EndToEndModel, self).__init__()
        self.gnn_models = nn.ModuleList(gnn_models)
        self.attn_combiners = nn.ModuleList(attn_combiners)

    def forward(self, graphs, node_embeddings, combination_strategies):
        embeddings = []
        for i, gnn_model in enumerate(self.gnn_models):
            embedding = gnn_model(graphs[i], node_embeddings[i])
            embeddings.append(embedding)

        combined_embeddings = []
        for item_idx, strategy in enumerate(combination_strategies):
            selected_embeddings = []
            for graph_idx, node_indices in strategy.items():
                selected_embeddings.append(embeddings[graph_idx][node_indices])
            selected_embeddings = torch.cat(selected_embeddings, dim=0)
            combined_embedding = self.attn_combiners[item_idx](selected_embeddings)
            combined_embeddings.append(combined_embedding)

        combined_embeddings = torch.stack(combined_embeddings, dim=0)
        return combined_embeddings

# 相似性推荐模型，使用MLP
class MLPRecommendation(nn.Module):
    def __init__(self, embedding_dim):
        super(MLPRecommendation, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, item_embedding, candidate_embedding):
        x = torch.cat([item_embedding, candidate_embedding], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 自定义数据集类
class PairDataset(Dataset):
    def __init__(self, combined_embeddings, pairs, labels):
        self.combined_embeddings = combined_embeddings
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item_id, candidate_id = self.pairs[idx]
        label = self.labels[idx]
        item_embedding = self.combined_embeddings[item_id]
        candidate_embedding = self.combined_embeddings[candidate_id]
        return item_embedding, candidate_embedding, label

# 初始化图和模型
embedding_dim = 16

# 假设我们有 n 个不同的图
n = 3  # 例如，有 3 个图
graphs = [
    dgl.graph(([0, 1], [1, 2]), num_nodes=3),  # 图1有3个节点
    dgl.graph(([0, 2], [2, 1]), num_nodes=5),  # 图2有5个节点
    dgl.graph(([0, 3], [3, 2]), num_nodes=4)   # 图3有4个节点
]

# 为每个图生成节点嵌入
node_embedding_layers = []
node_embeddings = []
for g in graphs:
    embedding_layer = nn.Embedding(g.number_of_nodes(), embedding_dim)
    node_embedding_layers.append(embedding_layer)
    node_embeddings.append(embedding_layer(torch.arange(g.number_of_nodes())))

# 初始化 GNN 模型列表
gnn_models = [GraphSAGE(in_feats=embedding_dim, h_feats=embedding_dim) for _ in range(n)]

# 初始化 SelfAttentionCombiner，每个物品一个
attn_combiners = [SelfAttentionCombiner(embedding_dim) for _ in range(len(graphs))]

# 构建端到端模型
end_to_end_model = EndToEndModel(gnn_models, attn_combiners)

# 定义组合策略
# 组合策略现在是一个字典，每个字典项代表一个图的索引及其对应的节点索引
combination_strategies = [
    {0: [1], 1: [3], 2: [2]},  # 物品0: 从图1取节点1、图2取节点3、图3取节点2
    {0: [2], 1: [4]},           # 物品1: 从图1取节点2、图2取节点4，不从图3中取节点
    {1: [0, 2], 2: [1]}         # 物品2: 从图2取节点0和2、图3取节点1，不从图1中取节点
]

# 生成组合后的物品嵌入
combined_embeddings = end_to_end_model(graphs, node_embeddings, combination_strategies)

# 初始化相似性推荐模型（MLP）
similarity_model = MLPRecommendation(embedding_dim)

# 假设正样本和负样本的物品对，以及它们的标签
positive_pairs = [(0, 1), (1, 2)]  # 正样本：物品0-物品1, 物品1-物品2
negative_pairs = [(0, 2), (1, 0)]  # 负样本：物品0-物品2, 物品1-物品0
labels = torch.tensor([1, 1, 0, 0], dtype=torch.float32)  # 正样本label为1，负样本label为0

# 创建数据集和DataLoader
dataset = PairDataset(combined_embeddings, positive_pairs + negative_pairs, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 训练模型
optimizer = torch.optim.Adam(list(end_to_end_model.parameters()) + list(similarity_model.parameters()), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()  # 用于二分类的交叉熵损失

# 开始训练
num_epochs = 100
for epoch in range(num_epochs):
    end_to_end_model.train()
    similarity_model.train()

    for item_embedding, candidate_embedding, label in dataloader:
        # 计算MLP输出
        outputs = similarity_model(item_embedding, candidate_embedding).squeeze()

        # 计算损失
        loss = loss_fn(outputs, label)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# 输出最终的损失
print(f'Final Loss: {loss.item()}')

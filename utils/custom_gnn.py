import dgl
import dgl.nn.pytorch as dgl_nn
from dgl.data.utils import load_info
import torch
import torch.nn as nn


# 设定运算设备
device = 'cpu'


# 定义模型
class GraphAttentionModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphAttentionModel, self).__init__()
        self.conv1 = dgl_nn.GATConv(in_feats=in_feats,
                                    out_feats=hidden_feats,
                                    num_heads=1,
                                    allow_zero_in_degree=True
                                    ).to(device)
        self.conv2 = dgl_nn.GATConv(in_feats=hidden_feats,
                                    out_feats=out_feats,
                                    num_heads=1,
                                    allow_zero_in_degree=True
                                    ).to(device)

    def forward(self, g, features, edge_weights):
        h = self.conv1(g, features)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h


# 初始化
# 初始化模型
model = GraphAttentionModel(in_feats=1, hidden_feats=16, out_feats=8).to(device)
# 使用损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 读入图
graphs = load_info(f"../dataset/processed/graph/graphs.bin")
for feature, graph in graphs.items():
    # 将节点特征和边属性移动到 GPU 上
    graph.ndata['index'] = graph.ndata['index'].to(device)
    graph.edata['weight'] = graph.edata['weight'].to(device)
    graph = dgl.add_self_loop(graph)
    labels = torch.arange(graph.num_nodes()).to(device)
    print(feature)
    # 训练模型
    for epoch in range(100):
        logits = model(graph, graph.ndata['index'].float(), graph.edata['weight'].unsqueeze(1).float())  # 将边权重作为注意力权重
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch {}, Loss: {:.4f}'.format(epoch, loss.item()))
    print('-----------------------------------------------------------------------------------------------------------')
    # 保存模型训练好的参数









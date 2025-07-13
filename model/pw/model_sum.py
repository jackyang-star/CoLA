import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import SAGEConv, GraphConv, GATConv


class GraphSAGE(nn.Module):
    def __init__(self, embedding_dim, gnn_layer_num):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        # 创建多个图卷积层
        for _ in range(gnn_layer_num):
            self.layers.append(SAGEConv(embedding_dim, embedding_dim, aggregator_type='mean'))

    def forward(self, g, inputs):
        h = inputs
        # 遍历所有层
        for i, layer in enumerate(self.layers):
            h = layer(g, h, edge_weight=g.edata['weight'])
            # 如果不是最后一层，应用ReLU
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h


class GCN(nn.Module):
    def __init__(self, embedding_dim, gnn_layer_num):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(gnn_layer_num):
            self.layers.append(GraphConv(embedding_dim, embedding_dim))

    def forward(self, g, inputs):
        # address the 0-in-degree node
        g = dgl.add_self_loop(g)

        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h


class GAT(nn.Module):
    def __init__(self, embedding_dim, gnn_layer_num):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()

        for _ in range(gnn_layer_num):
            self.layers.append(GATConv(embedding_dim, embedding_dim, 1))

    def forward(self, g, inputs):
        g = dgl.add_self_loop(g)
        h = inputs

        for i, layer in enumerate(self.layers):
            h = layer(g, h).flatten(1)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttention, self).__init__()
        self.emb_dim = embedding_dim
        self.Q_linear = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.K_linear = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.V_linear = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

    def forward(self, embeddings):
        query = self.Q_linear(embeddings)
        key = self.K_linear(embeddings)
        value = self.V_linear(embeddings)
        attention_scores = (torch.matmul(query, key.transpose(-2, -1)) /
                            torch.sqrt(torch.tensor(self.emb_dim, dtype=torch.float64)))  # emb_dim比较大的时候除以sqrt(emb_dim)比较好，例如emb_dim=512时
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_result = torch.matmul(attention_weights, value)
        return attention_result


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionLayer, self).__init__()
        self.self_attention = SelfAttention(embedding_dim)

    def forward(self, embeddings):
        # self-attention
        sa_embeddings = self.self_attention(embeddings)

        return sa_embeddings


class Aggregator(nn.Module):
    def __init__(self, embedding_dim, combiner_layer_num):
        super(Aggregator, self).__init__()
        layers1 = [AttentionLayer(embedding_dim) for _ in range(combiner_layer_num)]
        layers2 = [AttentionLayer(embedding_dim) for _ in range(combiner_layer_num)]
        self.layers1 = nn.ModuleList(layers1)
        self.layers2 = nn.ModuleList(layers2)

    def forward(self, embeddings):
        output0 = embeddings

        # attention method
        # for layer in self.layers1:
        #     output0 = layer(output0)
        # sum method
        output1 = output0.sum(dim=-2)
        # # flatten method
        # flattened_output1 = output0.view(output0.size(0), output0.size(1), -1)

        # attention method
        # for layer in self.layers2:
        #     output1 = layer(output1)
        # sum method
        output2 = output1.sum(dim=-2)
        # flatten method
        # flattened_output2 = flattened_output1.view(flattened_output1.size(0), -1)

        return output2


class Predictor(nn.Module):
    def __init__(self, embedding_dim):
        super(Predictor, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, combined_query_embedding, combined_cand_embedding):
        x = torch.cat((combined_query_embedding, combined_cand_embedding), dim=-1)
        complementary = self.linear(x)
        return complementary.squeeze()


class MVCG(nn.Module):
    def __init__(self, node_nums, strategies, device, embedding_dim, combiner_layer_num, gnn_layer_num):
        super(MVCG, self).__init__()
        self.device = device
        self.strategies = strategies
        self.node_nums = node_nums
        self.embedding_dim = embedding_dim

        self.graphsage_model0 = GraphSAGE(embedding_dim, gnn_layer_num)
        self.gcn_model0 = GCN(embedding_dim, gnn_layer_num)
        self.gat_model0 = GAT(embedding_dim, gnn_layer_num)
        self.node_embedding0 = nn.Embedding(node_nums[0], embedding_dim)
        self.graphsage_model1 = GraphSAGE(embedding_dim, gnn_layer_num)
        self.gcn_model1 = GCN(embedding_dim, gnn_layer_num)
        self.gat_model1 = GAT(embedding_dim, gnn_layer_num)
        self.node_embedding1 = nn.Embedding(node_nums[1], embedding_dim)
        self.graphsage_model2 = GraphSAGE(embedding_dim, gnn_layer_num)
        self.gcn_model2 = GCN(embedding_dim, gnn_layer_num)
        self.gat_model2 = GAT(embedding_dim, gnn_layer_num)
        self.node_embedding2 = nn.Embedding(node_nums[2], embedding_dim)
        self.graphsage_model3 = GraphSAGE(embedding_dim, gnn_layer_num)
        self.gcn_model3 = GCN(embedding_dim, gnn_layer_num)
        self.gat_model3 = GAT(embedding_dim, gnn_layer_num)
        self.node_embedding3 = nn.Embedding(node_nums[3], embedding_dim)
        # self.graphsage_model4 = GraphSAGE(embedding_dim, gcn_layer_num)
        # self.gcn_model4 = GCN(embedding_dim, gcn_layer_num)
        # self.node_embedding4 = nn.Embedding(node_nums[3], embedding_dim)
        # self.graphsage_model5 = GraphSAGE(embedding_dim, gcn_layer_num)
        # self.gcn_model5 = GCN(embedding_dim, gcn_layer_num)
        # self.node_embedding5 = nn.Embedding(node_nums[3], embedding_dim)
        # self.graphsage_model6 = GraphSAGE(embedding_dim, gcn_layer_num)
        # self.gcn_model6 = GCN(embedding_dim, gcn_layer_num)
        # self.node_embedding6 = nn.Embedding(node_nums[3], embedding_dim)
        # self.graphsage_model7 = GraphSAGE(embedding_dim, gcn_layer_num)
        # self.gcn_model7 = GCN(embedding_dim, gcn_layer_num)
        # self.node_embedding7 = nn.Embedding(node_nums[3], embedding_dim)
        # self.graphsage_model8 = GraphSAGE(embedding_dim, gcn_layer_num)
        # self.gcn_model8 = GCN(embedding_dim, gcn_layer_num)
        # self.node_embedding8 = nn.Embedding(node_nums[3], embedding_dim)
        # self.graphsage_model9 = GraphSAGE(embedding_dim, gcn_layer_num)
        # self.gcn_model9 = GCN(embedding_dim, gcn_layer_num)
        # self.node_embedding9 = nn.Embedding(node_nums[3], embedding_dim)
        self.combiner = Aggregator(embedding_dim, combiner_layer_num)
        # self.predictor_model = Predictor(embedding_dim)

    def padding_embeddings(self, batch_item, embeddings, feature_length):
        batch_item_embeddings = []
        for item in batch_item:
            # 初始化item_embeddings
            graph_num = len(embeddings)
            item_embeddings = [None] * graph_num
            # 在 feature 维度进行padding
            for graph_idx, node_indices in item.items():
                selected_embeddings = embeddings[graph_idx][node_indices].to(self.device)
                padding_size_nodes = feature_length - selected_embeddings.size(0)  # 计算节点维度的 padding 大小
                if padding_size_nodes > 0:
                    pad_nodes = (0, 0, 0, padding_size_nodes)  # 在节点维度（第一维）进行 padding
                    padded_embeddings = F.pad(selected_embeddings, pad_nodes)
                else:
                    padded_embeddings = selected_embeddings
                item_embeddings[graph_idx] = padded_embeddings
            # 在 graph 维度进行padding
            for i in range(graph_num):
                if item_embeddings[i] is None:
                    item_embeddings[i] = torch.zeros(feature_length, self.embedding_dim).to(self.device)  # 如果某个图没有嵌入，填充全零张量
            item_embeddings = torch.stack(item_embeddings)
            batch_item_embeddings.append(item_embeddings)
        batch_item_embeddings = torch.stack(batch_item_embeddings).to(self.device)
        return batch_item_embeddings

    def forward(self, graphs, batch_candidate, max_feature_length):  # query和candidate是组合策略，是字典形式
        # 生成node embeddings
        embeddings = []
        # use GraphSAGE to encode nodes
        embeddings.append(self.graphsage_model0(graphs[0], self.node_embedding0(torch.arange(self.node_nums[0]).to(self.device))))
        embeddings.append(self.graphsage_model1(graphs[1], self.node_embedding1(torch.arange(self.node_nums[1]).to(self.device))))
        embeddings.append(self.graphsage_model2(graphs[2], self.node_embedding2(torch.arange(self.node_nums[2]).to(self.device))))
        embeddings.append(self.graphsage_model3(graphs[3], self.node_embedding3(torch.arange(self.node_nums[3]).to(self.device))))
        # embeddings.append(self.graphsage_model4(graphs[4], self.node_embedding4(torch.arange(self.node_nums[4]).to(self.device))))
        # embeddings.append(self.graphsage_model5(graphs[5], self.node_embedding5(torch.arange(self.node_nums[5]).to(self.device))))
        # embeddings.append(self.graphsage_model6(graphs[6], self.node_embedding6(torch.arange(self.node_nums[6]).to(self.device))))
        # embeddings.append(self.graphsage_model7(graphs[7], self.node_embedding7(torch.arange(self.node_nums[7]).to(self.device))))
        # embeddings.append(self.graphsage_model8(graphs[8], self.node_embedding8(torch.arange(self.node_nums[8]).to(self.device))))
        # embeddings.append(self.graphsage_model9(graphs[9], self.node_embedding9(torch.arange(self.node_nums[9]).to(self.device))))
        # use GCN to encode nodes
        # embeddings.append(self.gcn_model0(graphs[0], self.node_embedding0(torch.arange(self.node_nums[0]).to(self.device))))
        # embeddings.append(self.gcn_model1(graphs[1], self.node_embedding1(torch.arange(self.node_nums[1]).to(self.device))))
        # embeddings.append(self.gcn_model2(graphs[2], self.node_embedding2(torch.arange(self.node_nums[2]).to(self.device))))
        # embeddings.append(self.gcn_model3(graphs[3], self.node_embedding3(torch.arange(self.node_nums[3]).to(self.device))))
        # use GAT to encode nodes
        # embeddings.append(self.gat_model0(graphs[0], self.node_embedding0(torch.arange(self.node_nums[0]).to(self.device))))
        # embeddings.append(self.gat_model1(graphs[1], self.node_embedding1(torch.arange(self.node_nums[1]).to(self.device))))
        # embeddings.append(self.gat_model2(graphs[2], self.node_embedding2(torch.arange(self.node_nums[2]).to(self.device))))
        # embeddings.append(self.gat_model3(graphs[3], self.node_embedding3(torch.arange(self.node_nums[3]).to(self.device))))
        # use WGAT to encode nodes

        # batch_query_strategy = []
        batch_candidate_strategy = []
        # for query in batch_query:
        #     batch_query_strategy.append(self.strategies[query])
        if batch_candidate.dim() > 0:
            for candidate in batch_candidate:
                batch_candidate_strategy.append(self.strategies[candidate])
        else:
            batch_candidate_strategy.append(self.strategies[batch_candidate])
        # 组合node embeddings
        # batch_query_embeddings = self.padding_embeddings(batch_query_strategy, embeddings, max_feature_length)
        # batch_combined_query_embedding = self.combiner(batch_query_embeddings)

        batch_cand_embeddings = self.padding_embeddings(batch_candidate_strategy, embeddings, max_feature_length)
        batch_combined_cand_embedding = self.combiner(batch_cand_embeddings)

        # 计算互补得分
        # batch_complementary_score = self.predictor_model(batch_combined_query_embedding, batch_combined_cand_embedding)

        return batch_combined_cand_embedding


class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.data = data
        self.query_data = [item[0] for item in data]
        self.cand_data = [item[1] for item in data]
        self.label_data = [item[2] for item in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query_item = self.query_data[idx]
        cand_item = self.cand_data[idx]
        label = self.label_data[idx]
        return query_item, cand_item, label


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, path='checkpoint.pt'):
        self.patience = patience  # 容忍的验证集性能连续下降的次数
        self.delta = delta  # 控制是否认为性能提升
        self.verbose = verbose  # 是否输出详细信息
        self.path = path  # 保存模型参数的路径
        self.counter = 0  # 计数器，记录验证集性能连续下降的次数
        self.best_score = None  # 最佳验证集性能
        self.early_stop = False  # 是否停止训练标志

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            # self.save_checkpoint(model)
        elif val_loss >= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            # self.save_checkpoint(model)

    def save_checkpoint(self, model):
        if self.verbose:
            print(f'Saving model parameters to {self.path}')
        torch.save(model.state_dict(), self.path)


class TeeOutput:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # 确保内容立即写入

    def flush(self):
        for f in self.files:
            f.flush()










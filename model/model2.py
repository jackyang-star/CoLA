import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
from torch.utils.data import Dataset
import sys


class GraphSAGE(nn.Module):
    def __init__(self, embedding_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(embedding_dim, embedding_dim, aggregator_type='mean')
        self.conv2 = SAGEConv(embedding_dim, embedding_dim, aggregator_type='mean')

    def forward(self, g, inputs):
        h = self.conv1(g, inputs, edge_weight=g.edata['weight'])  # 在计算中将权重转换为 float
        h = F.relu(h)
        h = self.conv2(g, h, edge_weight=g.edata['weight'])  # 同上
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


class CombinerLayer(nn.Module):
    def __init__(self, embedding_dim, dropout_rate):
        super(CombinerLayer, self).__init__()
        self.self_attention = SelfAttention(embedding_dim)
        self.attention_dropout = nn.Dropout(dropout_rate)

    def forward(self, embeddings):
        # layer normalization

        # self-attention
        sa_embeddings = self.self_attention(embeddings)
        # dropout
        # sa_embeddings = self.attention_dropout(sa_embeddings)
        # residual
        # sa_embeddings = sa_embeddings + embeddings

        # layer normalization

        # feed forward

        # dropout

        # residual


        return sa_embeddings


class Combiner(nn.Module):
    def __init__(self, embedding_dim, dropout_rate, combine_layer_num):
        super(Combiner, self).__init__()
        layers1 = [CombinerLayer(embedding_dim, dropout_rate) for _ in range(combine_layer_num)]
        layers2 = [CombinerLayer(embedding_dim, dropout_rate) for _ in range(combine_layer_num)]
        self.layers1 = nn.ModuleList(layers1)
        self.layers2 = nn.ModuleList(layers2)

    def forward(self, embeddings):
        output0 = embeddings

        for layer in self.layers1:
            output0 = layer(output0)
        output1 = output0.sum(dim=-2)

        for layer in self.layers2:
            output1 = layer(output1)
        output2 = output1.sum(dim=-2)

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
    def __init__(self, dropout_rate, embedding_dim, node_nums, strategies, device, combine_layer_num):
        super(MVCG, self).__init__()
        self.device = device
        self.strategies = strategies
        self.node_nums = node_nums

        self.gnn_model0 = GraphSAGE(embedding_dim)
        self.node_embedding0 = nn.Embedding(node_nums[0], embedding_dim)
        self.gnn_model1 = GraphSAGE(embedding_dim)
        self.node_embedding1 = nn.Embedding(node_nums[1], embedding_dim)
        self.gnn_model2 = GraphSAGE(embedding_dim)
        self.node_embedding2 = nn.Embedding(node_nums[2], embedding_dim)
        self.gnn_model3 = GraphSAGE(embedding_dim)
        self.node_embedding3 = nn.Embedding(node_nums[3], embedding_dim)
        self.combiner = Combiner(embedding_dim, dropout_rate, combine_layer_num)
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
                    item_embeddings[i] = torch.zeros(feature_length, 64).to(self.device)  # 如果某个图没有嵌入，填充全零张量
            item_embeddings = torch.stack(item_embeddings)
            batch_item_embeddings.append(item_embeddings)
        batch_item_embeddings = torch.stack(batch_item_embeddings).to(self.device)
        return batch_item_embeddings

    def forward(self, graphs, batch_candidate, max_feature_length):  # query和candidate是组合策略，是字典形式
        # 生成node embeddings
        embeddings = []
        embeddings.append(self.gnn_model0(graphs[0], self.node_embedding0(torch.arange(self.node_nums[0]).to(self.device))))
        embeddings.append(self.gnn_model1(graphs[1], self.node_embedding1(torch.arange(self.node_nums[1]).to(self.device))))
        embeddings.append(self.gnn_model2(graphs[2], self.node_embedding2(torch.arange(self.node_nums[2]).to(self.device))))
        embeddings.append(self.gnn_model3(graphs[3], self.node_embedding3(torch.arange(self.node_nums[3]).to(self.device))))

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










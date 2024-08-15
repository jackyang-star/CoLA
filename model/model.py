import torch
import numpy as np
import dgl
from dgl.nn.pytorch import SAGEConv
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, adjacency_matrix, embedding_matrix):
        x = torch.matmul(adjacency_matrix, embedding_matrix)
        x = self.linear(x)
        return x


class GCN(nn.Module):
    def __init__(self, n_nodes, emb_dim, layer_num):
        super(GCN, self).__init__()
        self.n_nodes = n_nodes
        self.emb_dim = emb_dim
        self.layer_num = layer_num

        self.emb = nn.Embedding(self.n_nodes, self.emb_dim)
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(emb_dim, emb_dim))
        self.layers.append(GCNLayer(emb_dim, emb_dim))

    def forward(self, cm_array):
        fv_embeddings = self.emb.weight

        # 行归一化
        row_sums = cm_array.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 防止除以0
        normalized_cm = cm_array / row_sums
        # 行列归一化
        # row_sums = np.array(cm_array.sum(1))
        # d_inv_sqrt = np.power(row_sums, -0.5)
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 将无穷大的值设为 0
        # d_inv_sqrt_mat = np.diag(d_inv_sqrt)
        # normalized_cm = d_inv_sqrt_mat @ cm_array @ d_inv_sqrt_mat

        normalized_cm_tensor = torch.from_numpy(normalized_cm).type(torch.FloatTensor)
        # GCN
        # hidden_embedding = fv_embeddings
        # final_embedding = fv_embeddings
        # for i in range(self.layer_num):
        #     hidden_embedding = torch.matmul(normalized_cm, hidden_embedding)
        #     final_embedding = final_embedding + hidden_embedding / (i + 1)
        for i, layer in enumerate(self.layers):
            fv_embeddings = layer(normalized_cm_tensor, fv_embeddings)
            if i != self.layer_num - 1:
                fv_embeddings = F.relu(fv_embeddings)

        return fv_embeddings


# 定义GraphSAGE模型
class GraphSAGE(nn.Module):
    def __init__(self, num_nodes, feats_dim):
        super(GraphSAGE, self).__init__()
        self.embedding = nn.Embedding(num_nodes, feats_dim)
        self.conv1 = SAGEConv(feats_dim, feats_dim, 'mean')
        self.conv2 = SAGEConv(feats_dim, feats_dim, 'mean')

    def forward(self, g, node_ids):
        h = self.embedding(node_ids)
        h = self.conv1(g, h, edge_weight=g.edata['weight'].float())  # 在计算中将权重转换为 float
        h = F.relu(h)
        h = self.conv2(g, h, edge_weight=g.edata['weight'].float())  # 同上
        return h


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, dropout_rate, head_num, iftrain):
        super(AttentionLayer, self).__init__()
        self.emb_dim = embedding_dim
        self.head_num = head_num
        self.head_dim = embedding_dim // head_num
        assert self.head_dim * head_num == embedding_dim, "Embedding dimension must be divisible by head_num"
        self.iftrain = iftrain

        self.Q_linear = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.K_linear = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.V_linear = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        # self.fc_out = nn.Linear(self.emb_dim, self.emb_dim)  # 用于多头注意力
        self.attention_dropout = nn.Dropout(dropout_rate)

    # 对样本进行layer normalization
    def layer_norm(self, tensor, epsilon=1e-8):
        global mean, variance
        if self.iftrain:
            if len(tensor.shape) == 4:
                mean = torch.mean(tensor, dim=(1, 2, 3), keepdim=True)
                variance = torch.var(tensor, dim=(1, 2, 3), keepdim=True)
            if len(tensor.shape) == 3:
                mean = torch.mean(tensor, dim=(1, 2), keepdim=True)
                variance = torch.var(tensor, dim=(1, 2), keepdim=True)
        else:
            if len(tensor.shape) == 3:
                mean = torch.mean(tensor, dim=(0, 1, 2), keepdim=True)
                variance = torch.var(tensor, dim=(0, 1, 2), keepdim=True)
            if len(tensor.shape) == 2:
                mean = torch.mean(tensor, dim=(0, 1), keepdim=True)
                variance = torch.var(tensor, dim=(0, 1), keepdim=True)

        normalized_tensor = (tensor - mean) / torch.sqrt(variance + epsilon)
        return normalized_tensor

    # 对embedding进行layer normalization
    def layer_norm2(self, tensor, epsilon=1e-8):
        mean = torch.mean(tensor, dim=(-1), keepdim=True)
        variance = torch.var(tensor, dim=(-1), keepdim=True)
        normalized_tensor = (tensor - mean) / torch.sqrt(variance + epsilon)
        return normalized_tensor

    def attention(self, batch_embeddings):
        # 计算生成query, key, value
        query = self.Q_linear(batch_embeddings)
        key = self.K_linear(batch_embeddings)
        value = self.V_linear(batch_embeddings)
        attention_scores = (torch.matmul(query, key.transpose(-2, -1)) /
                            torch.sqrt(torch.tensor(self.emb_dim, dtype=torch.float64)))  # emb_dim比较大的时候除以sqrt(emb_dim)比较好，例如emb_dim=512时
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_embedding = torch.matmul(attention_weights, value)
        return attention_embedding

        # if batch_embeddings.dim() == 3:
        #     N, seq_length, emb_dim = batch_embeddings.shape
        #     batch_embeddings = batch_embeddings.unsqueeze(1)  # Add channel dimension
        #     added_channel_dim = True
        # else:
        #     N, C, seq_length, emb_dim = batch_embeddings.shape
        #     added_channel_dim = False
        #
        #     # 计算生成query, key, value
        # queries = self.Q_linear(batch_embeddings).view(N, C, seq_length, self.head_num, self.head_dim).transpose(2, 3)
        # keys = self.K_linear(batch_embeddings).view(N, C, seq_length, self.head_num, self.head_dim).transpose(2, 3)
        # values = self.V_linear(batch_embeddings).view(N, C, seq_length, self.head_num, self.head_dim).transpose(2, 3)
        #
        # attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(
        #     torch.tensor(self.head_dim, dtype=torch.float32))
        # attention_weights = torch.softmax(attention_scores, dim=-1)
        # attention_weights = self.attention_dropout(attention_weights)
        # attention_output = torch.matmul(attention_weights, values)
        #
        # attention_output = attention_output.transpose(2, 3).contiguous().view(N, C, seq_length, emb_dim)
        #
        # if added_channel_dim:
        #     attention_output = attention_output.squeeze(1)
        #
        # return attention_output

    def forward(self, batch_embeddings):
        # attention
        attention_embedding = self.attention(batch_embeddings)
        # attention_embedding = self.fc_out(attention_embedding)  # 用于多头注意力
        # dropout
        attention_embedding = self.attention_dropout(attention_embedding)
        # residual
        attention_embedding = attention_embedding + batch_embeddings
        # layer normalization(两种，一种对整个样本，一种对单个embedding)
        # attention_embedding = self.layer_norm(attention_embedding)
        attention_embedding = self.layer_norm2(attention_embedding)
        return attention_embedding


class FeedForwardLayer(nn.Module):
    def __init__(self, embedding_dim, iftrain):
        super(FeedForwardLayer, self).__init__()
        self.emb_dim = embedding_dim
        self.iftrain = iftrain

        self.ff_linear = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * 4),
            nn.Linear(self.emb_dim * 4, self.emb_dim)
        )

    # 对样本进行layer normalization
    def layer_norm(self, tensor, epsilon=1e-8):
        global mean, variance
        if self.iftrain:
            if len(tensor.shape) == 4:
                mean = torch.mean(tensor, dim=(1, 2, 3), keepdim=True)
                variance = torch.var(tensor, dim=(1, 2, 3), keepdim=True)
            if len(tensor.shape) == 3:
                mean = torch.mean(tensor, dim=(1, 2), keepdim=True)
                variance = torch.var(tensor, dim=(1, 2), keepdim=True)
        else:
            if len(tensor.shape) == 3:
                mean = torch.mean(tensor, dim=(0, 1, 2), keepdim=True)
                variance = torch.var(tensor, dim=(0, 1, 2), keepdim=True)
            if len(tensor.shape) == 2:
                mean = torch.mean(tensor, dim=(0, 1), keepdim=True)
                variance = torch.var(tensor, dim=(0, 1), keepdim=True)

        normalized_tensor = (tensor - mean) / torch.sqrt(variance + epsilon)
        return normalized_tensor

    # 对embedding进行layer normalization
    def layer_norm2(self, tensor, epsilon=1e-8):
        mean = torch.mean(tensor, dim=(-1), keepdim=True)
        variance = torch.var(tensor, dim=(-1), keepdim=True)
        normalized_tensor = (tensor - mean) / torch.sqrt(variance + epsilon)
        return normalized_tensor

    def forward(self, attention_embedding):
        # feedforward
        ff_embedding = self.ff_linear(attention_embedding)
        # residual
        ff_embedding = ff_embedding + attention_embedding
        # layer normalization
        # ff_embedding = self.layer_norm(ff_embedding)
        ff_embedding = self.layer_norm2(ff_embedding)
        return ff_embedding


class AggregateLayer(nn.Module):
    def __init__(self, embedding_dim, dropout_rate, head_num, iftrain):
        super(AggregateLayer, self).__init__()
        self.emb_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.head_num = head_num
        self.iftrain = iftrain

        self.attention = AttentionLayer(self.emb_dim, self.dropout_rate, self.head_num, self.iftrain)
        # self.feedforward = FeedForwardLayer(self.emb_dim, self.iftrain)

    def forward(self, batch_embedding):
        attention_embedding = self.attention(batch_embedding)
        # ff_embedding = self.feedforward(attention_embedding)

        # 池化
        global aggregated_embedding
        if self.iftrain:
            if attention_embedding.dim() == 4:
                aggregated_embedding = attention_embedding.sum(dim=-2)
            elif attention_embedding.dim() == 3:
                aggregated_embedding = attention_embedding.view(attention_embedding.shape[0], -1)
        else:
            if attention_embedding.dim() == 3:
                aggregated_embedding = attention_embedding.sum(dim=-2)
            elif attention_embedding.dim() == 2:
                aggregated_embedding = attention_embedding.view(-1)

        return aggregated_embedding

class Aggregator(nn.Module):
    def __init__(self, embedding_dim, dropout_rate, head_num, iftrain):
        super(Aggregator, self).__init__()
        self.emb_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.head_num = head_num

        self.layer1 = AggregateLayer(self.emb_dim, self.dropout_rate, self.head_num, iftrain)
        self.layer2 = AggregateLayer(self.emb_dim, self.dropout_rate, self.head_num, iftrain)

    def forward(self, inputs):
        aggregate_output1 = self.layer1(inputs)
        aggregate_output2 = self.layer2(aggregate_output1)
        return aggregate_output2


class MVCG(nn.Module):
    def __init__(self, embedding_dim, dropout_rate, head_num, iftrain):
        super(MVCG, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.head_num = head_num
        self.hidden_dim = embedding_dim

        self.aggregator = Aggregator(self.embedding_dim, self.dropout_rate, self.head_num, iftrain)
        self.linear = nn.Sequential(
            nn.Linear(self.embedding_dim * 8, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, query, candidate):
        query_api_emb = self.aggregator(query)
        candidate_api_emb = self.aggregator(candidate)

        x = torch.cat((query_api_emb, candidate_api_emb), dim=-1)
        complementary = self.linear(x)

        return complementary.squeeze()


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)  # 加载数据为NumPy数组并转换为PyTorch张量
        self.query_data = self.data[:, 0]
        self.cand_data = self.data[:, 1]
        self.label_data = self.data[:, 2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = torch.from_numpy(np.array(self.query_data[idx], dtype=np.float32))
        cand = torch.from_numpy(np.array(self.cand_data[idx], dtype=np.float32))
        label = torch.from_numpy(np.array(self.label_data[idx], dtype=np.float32))
        return (query, cand, label)


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, path='checkpoint.pt'):
        self.patience = patience  # 容忍的验证集性能连续下降的次数
        self.delta = delta  # 控制是否认为性能提升
        self.verbose = verbose  # 是否输出详细信息
        self.path = path  # 保存模型参数的路径
        self.counter = 0  # 计数器，记录验证集性能连续下降的次数
        self.best_score = None  # 最佳验证集性能
        self.early_stop = False  # 是否停止训练标志

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss >= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        if self.verbose:
            print(f'Saving model parameters to {self.path}')
        torch.save(model.state_dict(), self.path)










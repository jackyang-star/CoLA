import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, h_feats, aggregator_type='mean')

    def forward(self, g, inputs):
        h = self.conv1(g, inputs, edge_weight=g.edata['weight'])  # 在计算中将权重转换为 float
        h = F.relu(h)
        h = self.conv2(g, h, edge_weight=g.edata['weight'])  # 同上
        return h


class Combiner(nn.Module):
    def __init__(self):
        super(Combiner, self).__init__()

    def forward(self, embeddings):
        pass


class Predictor(nn.Module):
    def __init__(self, embedding_dim, h_dim, graph_num):
        super(Predictor, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim * graph_num, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )

    def forward(self, combined_query_embedding, combined_cand_embedding):
        x = torch.cat((combined_query_embedding, combined_cand_embedding), dim=-1)
        complementary = self.linear(x)
        return complementary.squeeze()


class MVCG(nn.Module):
    def __init__(self, gnn_models, attn_combiner, predictor_model, embedding_dim, node_nums):
        super(MVCG, self).__init__()
        self.node_embeddings = []
        for num in node_nums:
            embedding = nn.Embedding(num, embedding_dim)
            self.node_embeddings.append(embedding(torch.arange(num)))
        self.gnn_models = gnn_models
        self.attn_combiner = attn_combiner
        self.predictor_model = predictor_model

    def forward(self, graphs, query, candidate):
        # 生成node embeddings
        embeddings = []
        for i, gnn_model in enumerate(self.gnn_models):
            embedding = gnn_model(graphs[i], self.node_embeddings[i])
            embeddings.append(embedding)

        # 组合node embeddings
        query_embeddings = []
        for graph_idx, node_indices in query.items():
            query_embeddings.append(embeddings[graph_idx][node_indices])
        query_embeddings = torch.cat(query_embeddings, dim=0)
        combined_query_embedding = self.attn_combiners(query_embeddings)
        candidate_embeddings = []
        for graph_idx, node_indices in candidate.items():
            candidate_embeddings.append(embeddings[graph_idx][node_indices])
        candidate_embeddings = torch.cat(candidate_embeddings, dim=0)
        combined_cand_embedding = self.attn_combiners(candidate_embeddings)

        # 计算互补得分
        complementary_score = self.predictor_model(combined_query_embedding, combined_cand_embedding)

        return complementary_score










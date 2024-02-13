import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformers import DistilBertModel, DistilBertTokenizer

class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

class NodeAttnMap_2(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap_2, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        e = F.softmax(e, dim=-1)

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x

class GCN_2(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN_2, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        # self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = F.elu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x

class GCN_3(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN_3, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.gelu = nn.GELU()

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.gelu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x

class GraphConvolutionWithSkip(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        # ... (same as GraphConvolution definition)
        super(GraphConvolutionWithSkip, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output + input  # Skip connection

class GCNWithSkip(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        # ... (same as GCN definition with skip connections)
        super(GCNWithSkip, self).__init__()
        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        gcn_layer = GraphConvolution(channels[0], channels[1])
        self.gcn.append(gcn_layer)
        for i in range(1, len(channels) - 1):
            gcn_layer = GraphConvolutionWithSkip(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x

class GraphIsomorphismLayer_2(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphIsomorphismLayer_2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.weight_agg = Parameter(torch.FloatTensor(out_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_agg.size(1))
        self.weight_agg.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        neighbor_aggregation = torch.matmul(adj, support)
        agg = support + neighbor_aggregation
        # agg = torch.cat([support, neighbor_aggregation], dim=-1)  # Improved aggregation
        output = torch.matmul(agg, self.weight_agg)  # Use matrix multiplication instead of spmm
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GIN_2(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GIN_2, self).__init__()

        self.gin_layers = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gin_layer = GraphIsomorphismLayer_2(channels[i], channels[i + 1])
            self.gin_layers.append(gin_layer)

    def forward(self, x, adj):
        for i in range(len(self.gin_layers) - 1):
            x = self.leaky_relu(self.gin_layers[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gin_layers[-1](x, adj)

        return x

class GIN_4(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GIN_4, self).__init__()

        self.gin_layers = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]

        total_dim = 0
        for i in nhid:
            total_dim += i

        self.readout = nn.Linear(total_dim, noutput)
        for i in range(len(channels) - 2):
            gin_layer = GraphIsomorphismLayer_2(channels[i], channels[i + 1])
            self.gin_layers.append(gin_layer)

    def forward(self, x, adj):
        xs = []
        for i in range(len(self.gin_layers)):
            x = self.leaky_relu(self.gin_layers[i](x, adj))
            xs.append(x)
            # print(x.shape)
        x = torch.cat(xs, dim=-1)
        # print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gin_layers[-1](x, adj)
        x = self.readout(x)

        return x

class GIN_3(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GIN_3, self).__init__()

        self.gin_layers = nn.ModuleList()
        self.dropout = dropout
        self.elu = nn.ELU()

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gin_layer = GraphIsomorphismLayer_2(channels[i], channels[i + 1])
            self.gin_layers.append(gin_layer)

    def forward(self, x, adj):
        for i in range(len(self.gin_layers) - 1):
            x = self.elu(self.gin_layers[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gin_layers[-1](x, adj)

        return x

class GraphIsomorphismLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphIsomorphismLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.matmul(adj, support)  # Use matrix multiplication instead of spmm
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GIN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GIN, self).__init__()

        self.gin_layers = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gin_layer = GraphIsomorphismLayer(channels[i], channels[i + 1])
            self.gin_layers.append(gin_layer)

    def forward(self, x, adj):
        for i in range(len(self.gin_layers) - 1):
            x = self.leaky_relu(self.gin_layers[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gin_layers[-1](x, adj)

        return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.W = Parameter(torch.zeros((in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

class GAT(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout, alpha=0.2, nheads=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        # Multi-head attention layers
        self.attentions = [GraphAttentionLayer(ninput, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # Output layer
        self.out_att = GraphAttentionLayer(nhid * nheads, noutput, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return F.elu(x)
    
    
class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed


class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.leaky_relu(x)
        return x

class GatedFusion(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(GatedFusion, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        # 定义门控单元
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, user_embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, input1, input2):
        # 将两个输入进行拼接
        combined_input = torch.cat((input1, input2), dim=0)
        
        # 计算门控权重
        gate_weight = self.gate(combined_input)
        
        # 使用门控权重对输入进行加权融合
        fused_output = gate_weight * input1 + (1 - gate_weight) * input2
        
        return fused_output

class FuseEmbeddings_2(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings_2, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed_1 = nn.Linear(embed_dim, embed_dim)
        self.fuse_embed_2 = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed_1(torch.cat((user_embed, poi_embed), 0))
        x = self.leaky_relu(x)
        x = self.fuse_embed_2(x)
        x = self.leaky_relu(x)
        return x

class FuseEmbeddings_3(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings_3, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.elu = nn.ELU()

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.elu(x)
        return x

class FuseEmbeddingsBilinear(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddingsBilinear, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        # self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.fuse_embed = nn.Bilinear(in1_features=user_embed_dim, in2_features=poi_embed_dim, out_features=embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(user_embed, poi_embed)
        x = self.leaky_relu(x)
        return x
    
class FuseEmbeddingsBN(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddingsBN, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        # self.fuse_embed = nn.Bilinear(in1_features=user_embed_dim, in2_features=poi_embed_dim, out_features=embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x

class FuseEmbeddingsDrop(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim, dropout_rate=0.3):
        super(FuseEmbeddingsDrop, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        # self.bn = nn.BatchNorm1d(embed_dim)
        # self.fuse_embed = nn.Bilinear(in1_features=user_embed_dim, in2_features=poi_embed_dim, out_features=embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.dropout(x)
        x = self.leaky_relu(x)
        return x
    
class ConvolutionalFuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim, hidden_dim=128, kernel_size=3):
        super(ConvolutionalFuseEmbeddings, self).__init__()

        self.user_linear = nn.Linear(user_embed_dim, hidden_dim)
        self.poi_linear = nn.Linear(poi_embed_dim, hidden_dim)

        # 卷积层用于融合
        self.convolution = nn.Conv1d(in_channels=2, out_channels=hidden_dim, kernel_size=kernel_size, stride=1, padding=1)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        user_proj = self.leaky_relu(self.user_linear(user_embed))
        poi_proj = self.leaky_relu(self.poi_linear(poi_embed))

        # 在通道维度上拼接两个投影
        fused_embedding = torch.cat((user_proj.unsqueeze(2), poi_proj.unsqueeze(2)), dim=2)

        # 通过卷积层进行融合
        fused_embedding = self.convolution(fused_embedding)

        # 拍平结果，以便作为最终输出
        fused_embedding = fused_embedding.view(fused_embedding.size(0), -1)
        
        print(fused_embedding.shape)

        return fused_embedding
    
# class AttentionFuseEmbeddings(nn.Module):
#     def __init__(self, user_embed_dim, poi_embed_dim, hidden_dim=128):
#         super(AttentionFuseEmbeddings, self).__init__()

#         self.user_linear = nn.Linear(user_embed_dim, hidden_dim)
#         self.poi_linear = nn.Linear(poi_embed_dim, hidden_dim)

#         self.attention_weights = nn.Parameter(torch.rand(hidden_dim))
#         self.leaky_relu = nn.LeakyReLU(0.2)

#     def forward(self, user_embed, poi_embed):
#         user_proj = self.leaky_relu(self.user_linear(user_embed))
#         poi_proj = self.leaky_relu(self.poi_linear(poi_embed))

#         # 计算注意力分数
#         # attention_scores = F.softmax(torch.matmul(user_proj * poi_proj, self.attention_weights), dim=0)
        
#         # 计算注意力分数
#         attention_scores = F.softmax(torch.sum(user_proj.T @ poi_proj, dim=1), dim=0)

#         # 扩展注意力权重，以便与投影相乘
#         attention_scores = attention_scores.unsqueeze(1).expand_as(user_proj)

#         # 加权融合
#         fused_embedding = attention_scores * user_proj + (1 - attention_scores) * poi_proj
        
#         print(attention_scores.shape)

#         # 加权融合
#         fused_embedding = attention_scores[0].item() * user_proj + attention_scores[1].item() * poi_proj

#         return fused_embedding


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(LearnablePositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        position = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.embedding(position)
        return self.dropout(x)

class LearnablePositionalEncoding_2(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(LearnablePositionalEncoding_2, self).__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        position = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.embedding(position)
        return x

class LearnableSPE(nn.Module):

  def __init__(self, d_model: int, d_hid: int, dropout: float = 0.1, max_len: int = 500):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    self.ffn = PositionwiseFeedForward(d_model, d_hid)
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)
  
  def forward(self, x):
    """
    Arguments:
        x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
    """
    self.pe_out = self.ffn(self.pe) # Add nn so that encoding becomes trainable
    x = x + self.pe_out[:x.size(0)]
    return self.dropout(x)

class PositionwiseFeedForward(nn.Module):

  def __init__(self, d_model, hidden, drop_prob=0.1):
    super(PositionwiseFeedForward, self).__init__()
    self.linear1 = nn.Linear(d_model, hidden)
    self.linear2 = nn.Linear(hidden, d_model)
    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(p=drop_prob)
  
  def forward(self, x):
    x = self.linear1(x)
    x = self.sigmoid(x)
    x = self.dropout(x)
    x = self.linear2(x)
    return x

class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout=0.1)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)
        self.decoder_cat.bias.data.zero_()
        self.decoder_cat.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat

class TransformerModel_2(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel_2, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat


class TransformerModel_3(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel_3, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = LearnablePositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat

class TransformerModel_4(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel_4, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = LearnablePositionalEncoding_2(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat

class TransformerModel_5(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel_5, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = LearnableSPE(d_model=embed_size, d_hid=128, dropout=0.1)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)
        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         nn.init.uniform_(param.data, -0.1, 0.1)
        #     elif 'bias' in name:
        #         param.data.zero_()

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat

class MyLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MyLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        decoded = self.linear(lstm_out)
        return decoded

class TransformerModel_6(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel_6, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model=embed_size, dropout=dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        # self.decoder_poi = nn.Linear(embed_size, num_poi)
        # self.decoder_time = nn.Linear(embed_size, 1)
        # self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.decoder_poi = MyLSTMModel(input_size=embed_size, hidden_size=128, num_layers=2, output_size=num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_cat = MyLSTMModel(input_size=embed_size, hidden_size=128, num_layers=2, output_size=num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        # initrange = 0.1
        # self.decoder_poi.bias.data.zero_()
        # self.decoder_poi.weight.data.uniform_(-initrange, initrange)
        pass
        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         nn.init.uniform_(param.data, -0.1, 0.1)
        #     elif 'bias' in name:
        #         param.data.zero_()

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat



class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weights = weights

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weights, ignore_index=-1)(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss

# class CustomDistilBertModel(nn.Module):
#     def __init__(self, num_poi, num_cat, embed_size, dropout=0.5):
#         super(CustomDistilBertModel, self).__init__()
#         self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')  # 使用预训练的 DistilBERT 模型
#         self.dropout = nn.Dropout(dropout)
#         self.decoder_poi = nn.Linear(768, num_poi)  # 调整输出维度
#         self.decoder_time = nn.Linear(768, 1)  # 调整输出维度
#         self.decoder_cat = nn.Linear(768, num_cat)  # 调整输出维度

#     def forward(self, input_ids, attention_mask):
#         outputs = self.distilbert(input_ids, attention_mask=attention_mask)
#         pooled_output = outputs['last_hidden_state'][:, 0, :]  # 取CLS token的表示
#         pooled_output = self.dropout(pooled_output)
#         out_poi = self.decoder_poi(pooled_output)
#         out_time = self.decoder_time(pooled_output)
#         out_cat = self.decoder_cat(pooled_output)
#         return out_poi, out_time, out_cat

import torch as th
from torch_scatter import scatter_mean, scatter_max
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GINConv
import copy

from networks import wordtransformer
from networks import posttransformer
import pickle

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
import sys
sys.path.append('./networks')

# 由于在实验的批处理中，输入是四维向量 [batch_size, statuses_len, word_max_len, embedding]，并且，我们设备的显存有限，因此，我们设计了轻量级bert，以适应该输入，理论上，BERT参数越多，模型的性能越好
class Statuses_Light_Bert(th.nn.Module):
    def __init__(self, dataname, word_embedding_dim, post_embedding_dim, word_head, post_head, word_dff, post_dff, word_droupout,
                 post_droupout, word_N, post_N, word_mask, post_mask, num_words,
                 batch_size, num_posts, out_dim):
        super(Statuses_Light_Bert, self).__init__()
        self.ws = pickle.load(open('./data/' + dataname + '_ws.pkl', 'rb'))
        self.c = copy.deepcopy
        self.num_words = num_words
        self.batch_size = batch_size
        self.num_posts = num_posts
        self.out_dim = out_dim
        # word
        self.word_embedding_dim = word_embedding_dim
        self.word_head = word_head
        self.word_dff = word_dff
        self.word_droupout = word_droupout
        self.word_N = word_N
        self.word_mask = word_mask
        self.word_embedding = wordtransformer.Embeddings(self.word_embedding_dim, len(self.ws))
        self.word_attention = wordtransformer.MultiHeadedAttention(self.word_head, self.word_embedding_dim)
        self.word_ff = wordtransformer.PositionwiseFeedForward(self.word_embedding_dim, self.word_dff,
                                                               self.word_droupout)
        self.word_layer = wordtransformer.EncoderLayer(self.word_embedding_dim, self.c(self.word_attention),
                                                       self.c(self.word_ff), self.word_droupout)
        self.word_en = wordtransformer.Encoder(self.word_layer, self.word_N)

        # post
        self.post_embedding_dim = post_embedding_dim
        self.post_head = post_head
        self.post_dff = post_dff
        self.post_droupout = post_droupout
        self.post_N = post_N
        self.post_mask = post_mask
        # self.post_embedding = posttransformer.Embeddings(self.post_embedding_dim, len(self.ws))
        self.post_attention = posttransformer.MultiHeadedAttention(self.post_head, self.post_embedding_dim)
        self.post_ff = posttransformer.PositionwiseFeedForward(self.post_embedding_dim, self.post_dff,
                                                               self.post_droupout)
        self.post_layer = posttransformer.EncoderLayer(self.post_embedding_dim, self.c(self.post_attention),
                                                       self.c(self.post_ff), self.post_droupout)
        self.post_en = posttransformer.Encoder(self.post_layer, self.post_N)
        self.post_linear = th.nn.Linear(self.post_embedding_dim, self.out_dim)

    def forward(self, data):
        """
        最终输出 （128*128的数据）

        """
        # word transformer
        history_statues = data.history_text  # (batch_size, num_posts, num_words)

        batch_size = history_statues.shape[0]

        x = self.word_embedding(history_statues)  # (batch_size, num_posts, num_words,embedding_dim)
        x = self.word_en(x, mask=None)  # (batch_size, num_posts, num_words,embedding_dim)
        # word_注意力最大化池化
        x = x.view(-1, self.num_words, self.word_embedding_dim)
        x = x.permute(0, 2, 1).contiguous()
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = x.view(batch_size, self.num_posts, self.word_embedding_dim)  # (batch_size,num_posts,embedding_dim)
        # post transformer
        x = self.post_en(x, mask=None)  # (batch_size,num_posts,embedding_dim) # (batch_size,num_posts,embedding_dim)
        x = F.adaptive_max_pool1d(x.permute(0, 2, 1).contiguous(), 1).squeeze(-1)  # (batch_size,embedding_dim)
        x = self.post_linear(x)  # (batch_size,out_dim)

        return x


class InteractionTree(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(InteractionTree, self).__init__()
        # in_fents 节点的特征数量   hid_feats 隐藏特征
        self.conv1 = GCNConv(in_feats, hid_feats)
        # 第二层要加上上一步输入的特征，所以输入为 隐藏特征 + 节点特征    out_feats输出特征
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        # x: [num_nodes, num_node_features] edge_index [2, num_edges] batch_size中所有图中的节点，最后使用data.batch  和 scatter_mean 区分
        # edge_index 与反向不同
        x, edge_index = data.x, data.edge_index
        # 作为 root1加强
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        # 作为 root2加强
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        # 把batch_size的所有图中节点的根节点特征 存入root_extend中
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # 把batch_size的所有图中节点的根节点特征 存入root_extend中
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        # 对应位置取平均，相当于一张图保留了一个节点的特征
        x = scatter_mean(x, data.batch, dim=0)

        return x


class Net(th.nn.Module):
    def __init__(self, dataname, batch_size, in_feats, hid_feats, out_feats, word_embedding_dim, post_embedding_dim, word_head, post_head,
                 word_dff, post_dff, word_droupout, post_droupout, word_N, post_N, num_words, num_posts, out_dim):
        super(Net, self).__init__()
        # model = Net(5000, 64, 64).to(device)
        self.StatusesEncoder = Statuses_Light_Bert(dataname=dataname, word_embedding_dim=word_embedding_dim, post_embedding_dim=post_embedding_dim, word_head=word_head, post_head=post_head,
                                               word_dff=word_dff, post_dff=post_dff, word_droupout=word_droupout, post_droupout=post_droupout,
                                               word_N=word_N, post_N=post_N, word_mask=None, post_mask=None, num_words=num_words,
                                               batch_size=batch_size, num_posts=num_posts, out_dim=out_dim)
        self.TDrumorGCN = InteractionTree(in_feats, hid_feats, out_feats)
        self.fc = th.nn.Linear((out_feats + hid_feats) * 2, 4)

    def forward(self, data):

        TN_x = self.StatusesEncoder(data)
        TD_x = self.TDrumorGCN(data)

        x = th.cat((TD_x, TN_x), 1)
        x_feature = x
        x = F.relu(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x, x_feature
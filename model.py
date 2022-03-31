import torch.nn as nn
from torch.nn.functional import cross_entropy, softmax, relu, leaky_relu
import numpy as np
import torch

LAYER_NUM = 3
"""
the input shape is [n_car, T, n_grid, embed_dim]
Therefore, for Transformer, the input dimension of each EncoderLayer in spatial transformer
can be expressed as [n_car, n_grid, embed_dim]
and the number of Encoder in spatial-temporal transformer is T
Therefore, the shape of multi-head attention layer in every encoder of spatial transformer
is [n_car, n_head, n_grid, head_dim]
And the input dimension of each EncoderLayer in temporal transformer is [n_car, T, embed_dim]
Therefore, the dimension of multi-head attention layer in every encoder of encoder of temporal
transformer is [n_car, n_head, T, head_dim]
"""


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    __constants__ = []

    def __init__(self, n_head, embed_dim, drop_rate, k_dim=None, v_dim=None):
        super().__init__()

        self.head_dim = embed_dim // n_head
        self.n_head = n_head
        self.embed_dim = embed_dim
        assert self.head_dim * self.n_head == self.embed_dim, "head_dim must be divisible by embed_dim"

        # 当Q、V、K的维度设置相等的时候，这三个变量不用
        self.q_dim = self.k_dim = k_dim if k_dim is not None else self.embed_dim
        self.v_dim = self.v_dim if v_dim is not None else self.embed_dim

        self.wq = nn.Linear(embed_dim, n_head * self.head_dim)  # [n_car, n_attentions, num_heads * head_dim]
        self.wk = nn.Linear(embed_dim, n_head * self.head_dim)
        self.wv = nn.Linear(embed_dim, n_head * self.head_dim)

        self.o_dense = nn.Linear(embed_dim, embed_dim)
        self.o_drop = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def sequence_mask(self, v_size, seq_len):
        mask = 1 - torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
        mask = mask.unsqueeze(0).expand(v_size, self.n_head, -1, -1).cuda()  # [B, L, L]
        return mask

    def forward(self, q, k, v, mask=None):
        residual = q
        batch_size = q.size(0)

        key = self.wk(k)  # []
        value = self.wv(v)  # []
        query = self.wq(q)  # []

        query = self.split_heads(query)  # [n_car, n_head, n_attentions, head_dim]
        key = self.split_heads(key)
        value = self.split_heads(value)

        # computer the attention
        context = self.scaled_dot_product_attention(query, key, value, mask)
        o = self.o_dense(context)  # [n_car, n_attentions, embed_dim]
        o = self.o_drop(o)

        o = self.layer_norm(residual + o)
        return o

    def split_heads(self, x):
        """
        划分头
        :param x:
        :return:
        """
        x = torch.reshape(x, (x.shape[0], x.shape[1], self.n_head, self.head_dim))
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        计算attention结果
        :param q: q矩阵
        :param k: k矩阵
        :param v: v矩阵

        :param mask: 是否mask处理
        :return:
        """
        dk = torch.tensor(k.shape[-1]).type(torch.float)
        score = torch.matmul(q, k.permute(0, 1, 3, 2)) / (torch.sqrt(dk) + 1e-8)  # [n, n_head, step, step]
        # print(score.shape)
        if mask is not None:
            mask = self.sequence_mask(score.shape[0], score.shape[2])
            # print(mask)
            score = score.masked_fill_(mask == 0, -np.inf)
        # print(score)
        self.attention = softmax(score, dim=-1)
        context = torch.matmul(self.attention, v)
        context = context.permute(0, 2, 1, 3)
        context = context.reshape((context.shape[0], context.shape[1], -1))
        return context


class PositionWiseFFN(nn.Module):
    """
    每个多头自注意力模型之间的前馈层
    """

    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        dff = embed_dim * 4
        # 每个前馈层共有两个全连接层
        self.l = nn.Linear(embed_dim, dff)
        self.o = nn.Linear(dff, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        o = relu(self.l(x))
        o = self.o(o)
        o = self.dropout(o)

        # residual + LayerNorm层
        o = self.layer_norm(x + o)
        return o  # [n_car, n_attentions, embed_dim]


class EncoderLayer(nn.Module):

    def __init__(self, n_head, emb_dim, drop_rate):
        super().__init__()
        # MultiHead + PositionWiseFFN 为一个Transformer的Encoder Layer
        self.mh = MultiHeadAttention(n_head, emb_dim, drop_rate)
        self.ffn = PositionWiseFFN(emb_dim)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, xz, mask):
        # xz: [n, step, emb_dim]
        context = self.mh(xz, xz, xz, mask)  # [n, step, emb_dim]
        o = self.ffn(context)
        return o


class Encoder(nn.Module):
    def __init__(self, n_head, emb_dim, drop_rate, n_layer):
        super().__init__()
        # 定义一个多层的Transformer Encoder
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(n_head, emb_dim, drop_rate) for _ in range(n_layer)]
        )

    def forward(self, xz, mask):
        for encoder in self.encoder_layers:
            xz = encoder(xz, mask)
        return xz


class PositionEmbedding(nn.Module):
    """
    位置编码信息，以及词汇信息编码
    """

    def __init__(
            self,
            max_len,  # 最大长度
            emb_dim,  # d_model
            input_dim,  # 输入的维度
            is_position  # 是否需要位置编码信息
    ):
        super().__init__()
        # np.expand_dims 扩充维度，第二个参数表示第二位扩充
        """
        np.expand_dims(np.arange([[1, 2, 3], [1, 2, 3]]), 1) ==> [[[1, 2, 3], [1, 2, 3]]]
        shape (1, 2) ==> shape (1, 1, 2)
        """
        # 位置信息编码
        pos = np.expand_dims(np.arange(max_len), 1)  # [max_len, 1]
        pe = pos / np.power(1000, 2 * np.expand_dims(np.arange(emb_dim), 0) / emb_dim)  # [max_len, emb_dim]
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = np.expand_dims(pe, 0)  # [1, max_len, emb_dim]

        self.is_position = is_position
        # 词汇信息编码
        self.pe = torch.from_numpy(pe).type(torch.float32)
        self.pe = self.pe.cuda()

        self.i = nn.Linear(input_dim, emb_dim * 2)
        self.o = nn.Linear(emb_dim * 2, emb_dim)

    def forward(self, x):
        i = relu(self.i(x))
        o = self.o(i)
        x_embed = o + self.pe if self.is_position else o  # [n, n_att dim]
        # 用词向量 + 位置向量 => 最后的输入
        return x_embed


class SpatialTransformer(nn.Module):
    """
    the spatial transformer
    """

    def __init__(self, n_input, n_grid, n_layer=3, emb_dim=64, n_head=4, drop_rate=0.1, padding_idx=0):
        super().__init__()
        self.padding_idx = torch.tensor(padding_idx)

        self.embedding = PositionEmbedding(n_grid, emb_dim, n_input, is_position=False)
        self.encoder = Encoder(n_head, emb_dim, drop_rate, n_layer)

    def forward(self, x):
        embedding = self.embedding(x)
        o = self.encoder(embedding, None)
        return o


class TemporalTransformer(nn.Module):
    """
    the temporal transformer
    """

    def __init__(self, n_input, t, n_layer=3, emb_dim=64, n_head=4, drop_rate=0.1):
        super().__init__()
        self.embedding = PositionEmbedding(t, emb_dim, n_input, is_position=True)
        self.encoder = Encoder(n_head, emb_dim, drop_rate, n_layer)

    def forward(self, x):
        embedding = self.embedding(x)
        o = self.encoder(embedding, True)
        return o


class SpatialTemporalTransformer(nn.Module):
    """
    the spatial-temporal transformer
    """

    def __init__(self, n_input, t, n_grid, n_spatial_layer=4, n_temporal_layer=3, emb_dim=64, n_head=4, drop_rate=0.1):
        super().__init__()

        # 定义时间spatial-temporal transformer
        # 其中每一个spatial-temporal transformer中都有t个spatial transformer和一个temporal transformer
        self.spatial = nn.ModuleList([
            SpatialTransformer(n_input, n_grid, n_spatial_layer, emb_dim, n_head, drop_rate) for _ in range(t)]
        )

        self.temporal = TemporalTransformer(emb_dim, t, n_temporal_layer, emb_dim, n_head, drop_rate)

        self.process_s = FullyConnectLayer(emb_dim * n_grid, emb_dim)
        self.process_t = FullyConnectLayer(emb_dim * n_grid, emb_dim)

        self.n_grid = n_grid
        self.t_windows = t
        self.emb_dim = emb_dim

    def forward(self, x):
        # the dimension of input is [n_car, t, n_grid, input_shape]
        x = torch.permute(x, (1, 0, 2, 3))  # [t, n_car, n_grid, input_shape]
        temporal_input = torch.zeros((self.t_windows, x.shape[1], self.emb_dim))  # [t, n_car, embed_dim]

        for i in range(x.shape[0]):
            o = self.spatial[i](x[i])  # [n_car, n_grid, input]
            o = torch.reshape(o, (o.shape[0], -1))
            o = self.process_s(o)
            temporal_input[i] = o

        # 3. 将每一个空间transformer的输出接入一个全连接层
        temporal_input = torch.permute(temporal_input, (1, 0, 2))  # [n_car, t, embed_dim]
        temporal_input = temporal_input.cuda()
        o = self.temporal(temporal_input)
        return o


class FullyConnectLayer(nn.Module):
    def __init__(self, input_dim, output_dim=300):
        super().__init__()

        self.i = nn.Linear(input_dim, 300)
        self.hidden = nn.ModuleList(
            [nn.Linear(300, 300) for _ in range(1)]
        )
        self.o = nn.Linear(300, output_dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        o = relu(self.i(x))
        # residual = o
        for v in self.hidden:
            o = relu(v(o))
            # o = self.drop(o)
        o = relu(self.o(o))
        return o


class ResultOperateModule(nn.Module):
    """
        Operate the input of the result from generator or real data
    """

    def __init__(self, input_shape, output_shape):
        super().__init__()
        # 由于输入维度大概在3，所以这里设置三个隐藏层
        self.i = nn.Linear(input_shape, 200)
        self.hidden_layer = nn.ModuleList(
            [nn.Linear(200, 200) for _ in range(3)]
        )
        self.o = nn.Linear(200, output_shape)

    def forward(self, x):
        o = relu(self.i(x))
        for v in self.hidden_layer:
            o = v(o)
        o = self.o(o)
        return o


class EnvironmentOperateModule(nn.Module):
    def __init__(self, n_input, t, n_grid, n_spatial_layer=3, n_temporal_layer=3, emb_dim=64, n_head=4, drop_rate=0.1):
        super().__init__()

        self.st = SpatialTemporalTransformer(n_input, t, n_grid, n_spatial_layer, n_temporal_layer, emb_dim, n_head,
                                             drop_rate)
        self.n_grid = n_grid
        self.t_windows = t
        self.emb_dim = emb_dim

        self.fc_i = nn.Linear(emb_dim, 200)
        self.fc_h = nn.ModuleList(
            [nn.Linear(200, 200) for _ in range(1)]
        )
        self.fc_o = nn.Linear(200, emb_dim)

    def forward(self, x):
        st = self.st(x)  # [n_car, t, embed_dim]
        st = torch.permute(st, (1, 0, 2))[-1]  # [n_car, embed_dim]

        o = relu(self.fc_i(st))  # [n_car, embed_dim]
        for v in self.fc_h:
            o = relu(v(o))
        o = self.fc_o(o)
        return o


class Generator(nn.Module):
    """
    Generator
    the Generator construction is:
        a spatial-temporal transformer
        a spatial transformer
        concatenate the above two output and send to a fully connect layer
    """

    def __init__(self, n_input, output_dim, t, n_grid, target='reg', n_spatial_layer=3, n_temporal_layer=3, emb_dim=32,
                 n_head=4, drop_rate=0.1):
        r"""
        initialize the class
        :param n_input: the input dimension
        :param output_dim: the output dimension
        :param t: time windows size
        :param n_grid: the grid size —— 21
        :param n_spatial_layer: the number of encoder layer
        :param emb_dim: the embedding dimension
        :param n_head: the number of head
        :param drop_rate: drop rate
        """
        super().__init__()
        self.target = target
        # spatial-temporal
        self.st = SpatialTemporalTransformer(n_input, t, n_grid, n_spatial_layer, n_temporal_layer, emb_dim, n_head,
                                             drop_rate)
        # spatial
        self.s = SpatialTransformer(n_input, n_grid, n_temporal_layer, emb_dim, n_head, drop_rate)
        self.process_st = FullyConnectLayer(emb_dim * t, emb_dim)
        self.process_s = FullyConnectLayer(emb_dim * n_grid, emb_dim)
        # fully connect layer
        self.fc = FullyConnectLayer(emb_dim * 3, 300)  # 需要将输出变成两种，一种是softmax，一种是
        self.embed = PositionEmbedding(emb_dim, emb_dim, input_dim=n_input, is_position=False)
        self.n_grid = n_grid
        self.c = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 3),
            nn.LogSoftmax(dim=1)
        )
        self.o = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, output_dim),
            # nn.ReLU()
        )

    def forward(self, x):
        # 输入维度 [n_car, t, n_grid, embed_dim]
        # 为了保证初始结果的差异,需要使用一个非常长的residual结构,将输入中的当前车辆的信息,经过embedding后加到时空encoder的输出中
        now_x = torch.permute(x, (1, 0, 2, 3))[-1]  # [n_car, n_grid, n_input]
        now_aim_x = torch.permute(now_x, (1, 0, 2))[self.n_grid // 2]  # [n_car, n_input]
        now_aim_x = now_aim_x.cuda()
        now_embed = self.embed(now_aim_x)
        s = self.s(torch.permute(x, (1, 0, 2, 3))[-1])  # [n_car, n_grid, embed_dim]
        st = self.st(x)  # [n_car, t, embed_dim]
        # print(s.shape, st.shape)
        # st = torch.sum(st, dim=1)  # 不取最后一秒的结果，而是将窗口中的所有输出累加在一起
        st = torch.reshape(st, (st.shape[0], -1))
        st = self.process_st(st)
        s = torch.reshape(s, (s.shape[0], -1))
        s = self.process_s(s)

        fc_input = torch.cat(
            (
                s,
                st,
                now_embed
            ),
            dim=-1
        )  # [n_car, embed_dim * 3]
        o = self.fc(fc_input)  # [n_car, 5]
        # 将FC layer的输出分别送入两个分支，一个是分类器，一个是回归器
        if self.target == 'reg':
            o = self.o(o)
        else:
            o = self.c(o)
        return o


class Discriminator(nn.Module):
    """
    Discriminator
        the discriminator construction is:
            a result operator
            a environment operator
            a fully connect layer
    """

    def __init__(self, n_input, r_input, t, n_grid, n_spatial_layer=3, n_temporal_layer=3, emb_dim=64, n_head=4,
                 drop_rate=0.1):
        super().__init__()
        self.result_operator = ResultOperateModule(r_input, emb_dim)

        self.st = SpatialTemporalTransformer(n_input, t, n_grid, n_spatial_layer, n_temporal_layer, emb_dim, n_head,
                                             drop_rate)
        self.n_grid = n_grid
        self.s = SpatialTransformer(n_input, n_grid, n_temporal_layer, emb_dim, n_head, drop_rate)
        self.embed = PositionEmbedding(emb_dim, emb_dim, input_dim=n_input, is_position=False)
        self.fc_decoder = nn.Sequential(
            nn.Linear(emb_dim * 3, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, emb_dim)
        )
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 200),
            nn.ReLU(True),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        """
        judge the source is g or real
        :param x:
            shape [n_car, t, n_grid, h]
        :param y:
            shape [n_car, 5]
        :return:
            fake or real
        """
        now_x = torch.permute(x, (1, 0, 2, 3))[-1]  # [n_car, n_grid, n_input]
        now_aim_x = torch.permute(now_x, (1, 0, 2))[self.n_grid // 2]  # [n_car, n_input]
        now_aim_x = now_aim_x.cuda()
        now_embed = self.embed(now_aim_x)
        r = self.result_operator(y)  # [n_car, embed_dim]
        # e = self.environment_operator(x)  # [n_car, embed_dim]
        st = self.st(x)  # [n_car, t, embed_dim]
        st = torch.permute(st, (1, 0, 2))[-1]
        s = self.s(torch.permute(x, (1, 0, 2, 3))[-1])
        input_decoder = torch.cat(
            (
                # en_embed,
                torch.permute(s, (1, 0, 2))[self.n_grid // 2],
                st,
                now_embed
            ),
            dim=-1
        )
        e = self.fc_decoder(input_decoder)
        t = torch.cat((r, e), 1)  # [n_car, embed_dim * 2]
        o = self.fc(t)
        return o

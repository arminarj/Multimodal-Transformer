import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys

# Code adapted from the fairseq repo.

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False, decorr=True):
        super().__init__()
        self.embed_dim = embed_dim
        # print(f'embed_dim : {embed_dim}')
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.num_projections = 4 if decorr else 3
        self.in_proj_weight = Parameter(torch.Tensor(self.num_projections * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(self.num_projections * embed_dim))
        self.out_proj = nn.Linear(2*embed_dim, embed_dim, bias=bias) if decorr else nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.decorr = decorr

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        # print('-----NEXT FORWARD LOOP-----')
        decorr = self.decorr
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            # print(f'----- SelfAttetion -----')
            # print(f'num_projections : {self.num_projections}')
            # print(f'decorr flag : {decorr}')
            q, k, v = self.in_proj_qkv(query)
            # print(f'q shape : {q.shape}'); print(f'k shape : {k.shape}'); print(f'v shape : {v.shape}')
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            if decorr:
                v_corr, v_decorr = self.in_proj_decorr_v(value)
                v = None
            else:
                v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            if decorr:
                v_corr = torch.cat([v_corr, self.bias_v.repeat(1, bsz, 1)])
                v_decorr = torch.cat([v_decorr, self.bias_v.repeat(1, bsz, 1)])
            else:
                v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
        # print(f'q shape : {q.shape}')
        # print(f'view size {tgt_len}, {bsz} , {self.num_heads}, {self.head_dim}')
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if decorr:
            if (v_corr is not None):
                v_corr = v_corr.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
                v_decorr = v_decorr.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        else:
            if v is not None:
                v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # print(f'v_corr with contiguous shape : {v_corr.shape}')
        # print(f'v_decorr with contiguous shape : {v_decorr.shape}')
        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            if decorr:
                v_corr = torch.cat([v_corr, v_decorr.new_zeros((v_corr.size(0), 1) + v_corr.size()[2:])], dim=1)
                v_decorr = torch.cat([v_decorr, v_decorr.new_zeros((v_decorr.size(0), 1) + v_decorr.size()[2:])], dim=1)
            else:
                v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
        
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False
        if decorr: 
            attn_weights_corr = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
            attn_weights_decorr = F.softmin(attn_weights.float(), dim=-1).type_as(attn_weights)
            # attn_weights = F.relu(attn_weights)
            # attn_weights = attn_weights / torch.max(attn_weights)
            attn_weights_corr = F.dropout(attn_weights_corr, p=self.attn_dropout, training=self.training)
            attn_weights_decorr = F.dropout(attn_weights_decorr, p=self.attn_dropout, training=self.training)
            # print(f'before V')
            # print(f'attn_weights_corr shape : {attn_weights_corr.shape}')
            # print(f'attn_weights_decorr shape : {attn_weights_decorr.shape}')
            # print(f'v_corr shape : {v_corr.shape}')
            # print(f'v_decorr shape : {v_decorr.shape}')

            attn_corr = torch.bmm(attn_weights_corr, v_corr)
            attn_decorr = torch.bmm(attn_weights_decorr, v_decorr)
            # print(f'----- after V -----')
            # print(f'corr shape : {attn_corr.shape}')
            # print(f'decorr shape : {attn_decorr.shape}')

            # attn = torch.cat([attn_corr, attn_decorr], dim=-1)
            # print(f'attention shape : {attn.shape}')
            assert list(attn_corr.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
            attn_corr = attn_corr.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            assert list(attn_decorr.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
            attn_decorr = attn_decorr.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            # print(f'----- after V -----')
            # print(f'corr shape : {attn_corr.shape}')
            # print(f'decorr shape : {attn_decorr.shape}')
            attn = torch.cat([attn_corr, attn_decorr], dim=-1)
            # print(f'attention shape : {attn.shape}')
            attn = self.out_proj(attn)
        else:
            attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
            # attn_weights = F.relu(attn_weights)
            # attn_weights = attn_weights / torch.max(attn_weights)
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

            attn = torch.bmm(attn_weights, v)
            assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        # print('-----NEXT Query mm -----')
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def in_proj_decorr_v(self, value):
        # print('-----NEXT decorr V-----')
        corr_v = self._in_proj(value, start=2 * self.embed_dim , end=3 * self.embed_dim)
        decorr_v = self._in_proj(value, start=3 * self.embed_dim)
        # print(f'corr V shape : {corr_v.shape}')
        # print(f'decorr V shape : {decorr_v.shape}')
        return corr_v, decorr_v

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        # if (out_start is None) and (out_end is None):
        #     weight = weight[start:end, :]
        # else:
        weight = weight[start:end, :]
        # print(f'start, end : {start}, {end}  ---- weight shape : {weight.shape}')
        if bias is not None:
            # if (bias_start is not None) or (bias_end is not None) :
            #     bias = bias[bias_start:bias_end] 
            # else:
            bias = bias[start:end]
        # print(f'weights shape : {weight.shape}')
        # print(f'bias shape : {bias.shape}')
        # print(f'input shape : {input.shape}')
        return F.linear(input, weight, bias)

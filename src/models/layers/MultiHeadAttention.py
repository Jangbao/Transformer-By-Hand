import torch.nn as nn
import torch
import logging
import models.layers.SelfAttention as self_attention

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    input: (batch_size, seq_len, d_model)
        ||
        N [x] self-attention: 将x进行自注意力计算，得到x (batch_size, seq_len, d_v)
        ||
        concat: 将x进行拼接，得到x (batch_size, seq_len, d_v * n_heads)
        ||
        linear: 将x进行线性变换，得到x (batch_size, seq_len, d_model)
        ||
    output: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model, d_k, d_v, n_heads):
        """
        Args:
            d_model: int, embedding dimension
            d_k: int, dimension of key
            d_v: int, dimension of value
            n_heads: int, number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads

        self.W_Qs = nn.ModuleList([])
        self.W_Ks = nn.ModuleList([])
        self.W_Vs = nn.ModuleList([])
        self.self_attentions = nn.ModuleList([])

        for _ in range(n_heads):
            self.W_Qs.append(nn.Linear(d_model, d_k, bias=True))
            self.W_Ks.append(nn.Linear(d_model, d_k, bias=True))
            self.W_Vs.append(nn.Linear(d_model, d_v, bias=True))
            self.self_attentions.append(self_attention.SelfAttention(d_model, d_k, d_v))  

        self.W_O = nn.Linear(d_v * n_heads, d_model, bias=True)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: (batch_size, seq_len, d_model)
            k: (batch_size, seq_len, d_model)
            v: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) (optional)
        Returns:
            multi_head_attention_output: (batch_size, seq_len, d_model)
        """
        # torch.cat 
        # 拼接时，除了 dim 指定的维度外，其他维度必须完全相同。
        # 拼接后，指定的维度长度是所有输入张量该维度长度之和。
        self_attention_outputs = []

        for i in range(self.n_heads):
            q_i = self.W_Qs[i](q)
            k_i = self.W_Ks[i](k)
            v_i = self.W_Vs[i](v)

            self_attention_outputs.append(self.self_attentions[i](q_i, k_i, v_i, mask))
        logging.debug(f"self_attention_outputs.shape: {self_attention_outputs[0].shape}")
        multi_head_attention_output = torch.cat(self_attention_outputs, dim=-1)
        logging.debug(f"multi_head_attention_output.shape: {multi_head_attention_output.shape}")
        multi_head_attention_output = self.W_O(multi_head_attention_output)
        logging.debug(f"multi_head_attention_output.shape: {multi_head_attention_output.shape}")
        return multi_head_attention_output
import torch.nn as nn

import models.layers.MultiHeadAttention as multi_head_attention
import models.layers.FFN as ffn
import models.layers.AddAndNorm as add_and_norm

class EncoderLayer(nn.Module):
    """
    Encoder Layer
    input: (batch_size, seq_len, d_model)
        ||    ||
        ||    MultiHeadAttention: 将x进行自注意力计算，得到x (batch_size, seq_len, d_model)
        ||    ||
        AddAndNorm: 将x和sub_layer_output相加，得到x (batch_size, seq_len, d_model)
        ||    ||
        ||    FFN: 将x进行前馈神经网络计算，得到x (batch_size, seq_len, d_model)
        ||    ||
        AddAndNorm: 将x和sub_layer_output相加，得到x (batch_size, seq_len, d_model)
        ||
    output: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model, d_k, d_v, n_heads, d_ff, dropout):
        """
        Args:
            d_model: int, embedding dimension
            d_k: int, dimension of key
            d_v: int, dimension of value
            n_heads: int, number of heads
            d_ff: int, feedforward dimension
            dropout: float, dropout rate
        """
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = multi_head_attention.MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.add_and_norm1 = add_and_norm.AddAndNorm(d_model, dropout)
        self.ffn = ffn.FFN(d_model, d_ff)
        self.add_and_norm2 = add_and_norm.AddAndNorm(d_model, dropout)


    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) (optional)
        Returns:
            x: (batch_size, seq_len, d_model)
        """
        x = self.add_and_norm1(x, self.multi_head_attention(q=x, k=x, v=x))
        x = self.add_and_norm2(x, self.ffn(x))
        return x
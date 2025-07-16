import torch.nn as nn
import torch
import logging

class SelfAttention(nn.Module):

    """
    Self Attention
    input: (batch_size, seq_len, d_model)
        ||
        matmul: 将q和k^t进行矩阵乘法，得到x (batch_size, seq_len, seq_len)
        ||
        scaled: 将x进行缩放，得到x (batch_size, seq_len, seq_len)
        ||
        dropout: 将x进行dropout，得到x (batch_size, seq_len, seq_len)
        ||
        softmax: 将x进行softmax，得到x (batch_size, seq_len, seq_len)
        ||
        matmul: 将x和v进行矩阵乘法，得到x (batch_size, seq_len, d_v)
        ||
    output: (batch_size, seq_len, d_v)
    """

    def __init__(self, d_model, d_k, d_v):
        """
        Args:
            d_model: int, embedding dimension
            d_k: int, dimension of key
            d_v: int, dimension of value
        """
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.scale = d_k ** -0.5
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, q, k, v, mask=None, e=1e-12):
        """
        Args:
            q: (batch_size, seq_len, d_model)
            k: (batch_size, seq_len, d_model)
            v: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) (optional)
            e: float, epsilon
        Returns:
            x: (batch_size, seq_len, d_v)
        """
        # transpose(-2,-1)
        # 对张量 x 的倒数第二个维度和最后一个维度交换。
        x = torch.matmul(q, k.transpose(-2, -1)) # (batch_size, seq_len, seq_len)
        x = x * self.scale # (batch_size, seq_len, seq_len)

        if mask is not None:
            x = x.masked_fill(mask == 0, -1e9)

        # 防止softmax出现nan
        x = x + e
        x = self.softmax(x) # (batch_size, seq_len, seq_len)
        x = torch.matmul(x, v) # (batch_size, seq_len, d_v)
        
        return x
    
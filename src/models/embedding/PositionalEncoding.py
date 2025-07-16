import torch.nn as nn
import torch
from torch.autograd import Variable
import logging

class PositionalEncoding(nn.Module):

    """
    Positional Encoding
    input: (batch_size, seq_len, d_model)
        ||
        positional encoding: 将输入的词索引转换为positional encoding向量 (batch_size, seq_len, d_model)
        ||
    output: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model, max_len, device):
        """
        Args:
            d_model: int, embedding dimension
            max_len: int, maximum length of input sequence
            device: str, device name
        """

        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model, device=device) # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float,device=device).unsqueeze(1) # (max_len, 1)
        # position: torch.arange(0, d_model, 2, dtype=torch.float) -> [0, 2, 4, 6, ...]

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float,device=device) * -(torch.log(torch.tensor(10000.0, device=device)) / d_model))   # (d_model/2)
        # div_term: e^ [-log(10000) * X / d_model]
        #         = {e^ [-log(10000)]} ^ X / d_model
        #         = (1/10000) ^ (X / d_model)
        #         = 1/10000 ^ (X / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.pe = pe.to(device)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            pe: (1, seq_len, d_model)
        """
        return Variable(self.pe[:, :x.size(1)], requires_grad=False)
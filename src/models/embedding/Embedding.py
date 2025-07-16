import torch.nn as nn
import math
import logging

class Embedding(nn.Module):

    """
    Embedding
    input: (batch_size, seq_len)
        ||
        embedding: 将输入的词索引转换为embedding向量 (batch_size, seq_len, d_model)
        ||
    output: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model, vocab_size):
        """
        Args:
            d_model: int, embedding dimension
            vocab_size: int, vocabulary size
        """
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # 定义一个embedding层，将输入的词索引转换为embedding向量
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len)
        Returns:
            x: (batch_size, seq_len, d_model)
        """

        # 为什么要乘以 math.sqrt(self.d_model)？
        # 如果不做缩放：
        # - embedding 和 positional encoding 数值范围可能不匹配（embedding 比 positional encoding 的幅度小）；
        # - positional encoding 对模型输出的影响就会过大。

        logging.debug(f"embedding.weight.device: {self.embedding.weight.device}")

        logging.debug(f"before Embedding forward, x.shape: {x.shape}")
        x = self.embedding(x) * math.sqrt(self.d_model)
        logging.debug(f"after Embedding forward, x.shape: {x.shape}")
        return x
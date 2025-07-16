import torch.nn as nn

class FFN(nn.Module):
    """
    Feedforward Neural Network
    input: (batch_size, seq_len, d_model)
        ||
        linear1: 将x进行线性变换，得到x (batch_size, seq_len, d_ff)
        ||
        relu: 将x进行ReLU激活，得到x (batch_size, seq_len, d_ff)
        ||
        linear2: 将x进行线性变换，得到x (batch_size, seq_len, d_model)
        ||
    output: (batch_size, seq_len, d_model)
    """

    def __init__(self,d_model,d_ff):
        """
        Args:
            d_model: int, embedding dimension
            d_ff: int, feedforward dimension
        """
        super(FFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # 定义两个线性层，一个用于线性变换，一个用于线性变换的逆变换
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)
        # 定义一个ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x: (batch_size, seq_len, d_model)
        """
        return self.linear2(self.relu(self.linear1(x)))

    
    
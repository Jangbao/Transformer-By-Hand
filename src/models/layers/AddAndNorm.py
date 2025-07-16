import torch.nn as nn

class AddAndNorm(nn.Module):
    """
    Add and Normalization
    input: (batch_size, seq_len, d_model)
        ||
        residual connection: 将sub_layer_output添加到x中 (batch_size, seq_len, d_model)
        ||
        layer normalization: 对x进行归一化 (batch_size, seq_len, d_model)
        ||
    output: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model, dropout=0.1):
        """
        Args:
            d_model: int, embedding dimension
            sub_layer_output: (batch_size, seq_len, d_model)
        """
        super(AddAndNorm, self).__init__()
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sub_layer_output):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            sub_layer_output: (batch_size, seq_len, d_model)
        Returns:
            x: (batch_size, seq_len, d_model)
        """
        # residual connection: 将sub_layer_output添加到x中
        # layer normalization: 对x进行归一化
        x = self.dropout(x)
        return self.layer_norm(x + sub_layer_output)
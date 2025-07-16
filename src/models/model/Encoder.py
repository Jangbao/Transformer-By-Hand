import torch.nn as nn
import logging
import models.embedding.Embedding as embedding
import models.embedding.PositionalEncoding as positional_encoding
import models.blocks.EncoderLayer as encoder_layer

class Encoder(nn.Module):
    """ 
    Encoder
    input: (batch_size, seq_len)
        ||
        Embedding: 将输入的词索引转换为embedding向量 (batch_size, seq_len, d_model)
            +
        PositionalEncoding: 将输入的词索引转换为positional encoding向量 (batch_size, seq_len, d_model)
        ||
        N x EncoderLayer: 执行N次EncoderLayer，得到输出 (batch_size, seq_len, d_model)
        ||
    output: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model, vocab_size, d_k, d_v, n_heads, d_ff, max_len, n_layers,dropout, device):
        """
        Args:
            d_model: int, embedding dimension
            vocab_size: int, vocabulary size
            d_k: int, dimension of key
            d_v: int, dimension of value
            n_heads: int, number of heads
            d_ff: int, feedforward dimension
            max_len: int, maximum length of input sequence
            n_layers: int, number of layers
            dropout: float, dropout rate
            device: str, device name
        """
        super(Encoder, self).__init__()
        self.embedding = embedding.Embedding(d_model, vocab_size)
        self.positional_encoding = positional_encoding.PositionalEncoding(d_model,  max_len, device)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([encoder_layer.EncoderLayer(d_model,d_k,d_v, n_heads,d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len)
            mask: (batch_size, seq_len, seq_len) (optional)
        Returns:
            x: (batch_size, seq_len, d_model)
        """
        logging.debug("running encoder embedding start...")
        x = self.embedding(x)
        logging.debug("running encoder embedding finished.")
        logging.debug("running encoder positional encoding start...")
        pe = self.positional_encoding(x)
        x = x + pe
        logging.debug("running encoder positional encoding finished.")
        logging.debug("running encoder dropout start...")
        x = self.dropout(x)
        logging.debug("running encoder dropout finished.")
        logging.debug("running encoder layers start...")
        # N x EncoderLayer
        for i, layer in enumerate(self.layers):
            logging.debug(f"running encoder layer {i} start...")
            x = layer(x)    
            logging.debug(f"running encoder layer {i} finished.") 
        logging.debug("running encoder layers finished.")
        return x
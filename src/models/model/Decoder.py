import torch.nn as nn
import logging
import models.embedding.Embedding as embedding
import models.embedding.PositionalEncoding as positional_encoding
import models.blocks.DecoderLayer as decoder_layer

class Decoder(nn.Module):
    """ 
    Decoder
    input: (batch_size, seq_len)
        ||
        Embedding: 将x进行embedding计算，得到x (batch_size, seq_len, d_model)
        +
        PositionalEncoding: 将x进行positional encoding计算，得到x (batch_size, seq_len, d_model)
        ||
        ||   Encoder-Output: 将encoder的输出作为decoder的输入 (batch_size, seq_len, d_model)
        ||   ||
        N x DecoderLayer: 将x进行decoder layer计算，得到x (batch_size, seq_len, d_model)
        ||
        Linear: 将x进行线性计算，得到x (batch_size, seq_len, vocab_size)
        ||
        Softmax: 将x进行softmax计算，得到probalities (batch_size, seq_len, vocab_size)
        ||
    output: (batch_size, seq_len, vocab_size)
    """
    def __init__(self, d_model, vocab_size, d_k, d_v, n_heads, max_len, d_ff, n_layers, dropout, device):
        """
        Args:
            d_model: int, embedding dimension
            vocab_size: int, vocabulary size
            d_k: int, dimension of key
            d_v: int, dimension of value
            n_heads: int, number of heads
            max_len: int, maximum length of input sequence
            d_ff: int, feedforward dimension
            n_layers: int, number of layers
            dropout: float, dropout rate
            device: str, device name
        """
        super(Decoder, self).__init__()
        self.embedding = embedding.Embedding(d_model, vocab_size)
        self.positional_encoding = positional_encoding.PositionalEncoding(d_model,max_len, device)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([decoder_layer.DecoderLayer(d_model, d_k, d_v, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, encoder_output, mask=None):

        logging.debug("running decoder embedding start...")
        x = self.embedding(x)
        logging.debug("running decoder embedding finished.")
        logging.debug("running decoder positional encoding start...")
        pe = self.positional_encoding(x)
        x = x + pe
        logging.debug("running decoder positional encoding finished.")
        logging.debug("running decoder dropout start...")
        x = self.dropout(x)
        logging.debug("running decoder dropout finished.")
        logging.debug("running decoder layers start...")
        for i, layer in enumerate(self.layers):
            logging.debug(f"running decoder layer {i} start...")
            x = layer(x, encoder_output, mask)
            logging.debug(f"running decoder layer {i} finished.")
        logging.debug("running decoder layers finished.")
        logging.debug("running decoder linear start...")
        x = self.linear(x)
        logging.debug("running decoder linear finished.")
        logging.debug("running decoder softmax start...")
        x = self.softmax(x)
        logging.debug("running decoder softmax finished.")
        return x
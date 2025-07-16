import torch.nn as nn
import torch
import logging
import models.model.Encoder as encoder
import models.model.Decoder as decoder

class Transformer(nn.Module):
    """
    Transformer
    input_seq: (batch_size, seq_len)
        ||
        Encoder: 将input_seq进行encoder计算，得到encoder_output (batch_size, seq_len, d_model)
        ||
        ||   output_seq: (batch_size, seq_len)
        ||   ||
        Decoder: 将output_seq和encoder_output进行decoder计算，得到decoder_output (batch_size, seq_len, d_model)
        ||
    final_output: (batch_size, seq_len, vocab_size)
    """

    def __init__(self, d_model, encoder_vocab_size, decoder_vocab_size, d_k, d_v, d_ff, n_heads,max_len, n_layers, dropout, device):
        """
        Args:
            d_model: int, embedding dimension
            encoder_vocab_size: int, vocabulary size
            decoder_vocab_size: int, vocabulary size
            d_k: int, dimension of key
            d_v: int, dimension of value
            d_ff: int, feedforward dimension
            n_heads: int, number of heads   
            max_len: int, maximum length of input sequence
            n_layers: int, number of layers
            dropout: float, dropout rate
            device: str, device name
        """
        super(Transformer, self).__init__()

        # encoder 和 decoder 的 vocab_size 可以相同，也可以不同
        # 场景	encoder vocab	decoder vocab	是否一样	为什么
        # 机器翻译（en → de）	英语词表大小	德语词表大小	不同	因为输入是英语，输出是德语，自然词表不同
        # 自编码器（自监督任务 / BERT）	相同	相同	一样	输入和输出都是同一语言
        self.encoder = encoder.Encoder(d_model, encoder_vocab_size, d_k, d_v, n_heads, d_ff, max_len, n_layers, dropout, device)
        self.decoder = decoder.Decoder(d_model, decoder_vocab_size, d_k, d_v, n_heads, d_ff, max_len, n_layers, dropout, device)
        self.device = device
        
    def forward(self, input_seq, output_seq):
        """
        Args:
            input_seq: (batch_size, seq_len)
            output_seq: (batch_size, seq_len)
        Returns:
            decoder_output: (batch_size, seq_len, vocab_size)
        """
        mask = self.create_tril_mask(output_seq).to(self.device)
        
        logging.debug("running encoder start...")
        encoder_output = self.encoder(input_seq, mask)
        logging.debug("running encoder finished.")
        logging.debug("running decoder start...")
        decoder_output = self.decoder(output_seq, encoder_output, mask)
        logging.debug("running decoder finished.")
        return decoder_output

    def create_tril_mask(self, output_seq):
        """
        Args:
            output_seq: (batch_size, seq_len)
        Returns:
            mask: (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len = output_seq.size()

        # torch.tril 是 PyTorch 中的一个函数，用来返回张量的下三角部分，其余部分设为 0。
        # 它的名字是：tril = triangular lower。
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.expand(batch_size,  seq_len, seq_len)
        return mask
from Config import *
from util.DataLoader import DataLoader
from util.Tokenizer import Tokenizer

tokenizer = Tokenizer()
loader = DataLoader(tokenize_de=tokenizer.tokenize_de, tokenize_en=tokenizer.tokenize_en,ext=('.de', '.en'), init_token='<sos>', eos_token='<eos>')

train_data, valid_data, test_data = loader.make_dataset()
loader.build_vocab(train_data, min_freq=2)
train_iter, valid_iter, test_iter = loader.make_iter(train_data, valid_data, test_data,
                                                     batch_size=batch_size,
                                                     device=device)

src_pad_idx = loader.source.vocab.stoi['<pad>']
trg_pad_idx = loader.target.vocab.stoi['<pad>']
trg_sos_idx = loader.target.vocab.stoi['<sos>']

enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)



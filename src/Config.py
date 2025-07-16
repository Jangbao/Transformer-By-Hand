import torch
import logging

#logging setting
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#folder setting
base_path = "/root/python_project/Transformer-By-Hand"
dataset_path = base_path + "/dataset/"
saved_path = base_path + "/saved/"
result_path = base_path + "/result/"

train_loss_path = result_path + "train_loss.txt"
test_loss_path = result_path + "test_loss.txt"
bleu_path = result_path + "bleu.txt"

# model parameter setting
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
d_model = 512
d_k = 64
d_v = 64
d_ff = 2048
n_heads = 8
max_len = 256
n_layers = 6
dropout = 0.1
num_epochs = 1000
batch_size = 128
learning_rate = 1e-5
weight_decay = 5e-4
adam_eps = 5e-9
factor = 0.9
patience = 10
warmup = 100
clip = 1.0
inf = float('inf')
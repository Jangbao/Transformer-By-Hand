# %%
#auto reload the code
%load_ext autoreload
%autoreload 2

# %%
#data initialization
from Data import *

# %%
#module import
import models.model.Transformer as transformer 
import time
import math
from torch.optim import Adam
from torch import nn, optim
from util.Bleu import *
from util.EpochTimer import *



# %%
#model initialization 
def init_weights(m):
    """
    初始化权重
    """
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight)
    elif hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias, 0)


model = transformer.Transformer(d_model=d_model, 
                                encoder_vocab_size=enc_voc_size,   
                                decoder_vocab_size=dec_voc_size,
                                d_k=d_k,
                                d_v=d_v,
                                d_ff=d_ff,
                                n_heads=n_heads, 
                                max_len=max_len, 
                                n_layers=n_layers, 
                                dropout=dropout, 
                                device=device).to(device)

model.apply(init_weights)

# 定义优化器
optimizer = Adam(params=model.parameters(),
                 lr=learning_rate,
                 weight_decay=weight_decay,
                 eps=adam_eps)
# 学习率衰减
# 这是 PyTorch 内置的 ReduceLROnPlateau，作用：
# 当监控的指标（比如验证集 loss）在若干个 epoch 内都没有好转时
# 自动把学习率乘以 factor（比如 0.1）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

# 定义损失函数
criterion = torch.nn.CrossEntropyLoss()

# %%
#train

def train(model, iterator, optimizer, criterion, clip):
    """
    训练模型
    Args:
        model: 模型
        iterator: 数据迭代器
        optimizer: 优化器
        criterion: 损失函数
        clip: 梯度裁剪
    """
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        logging.debug(f"iteration {i} src.shape: {src.shape}, trg.shape: {trg.shape}")

        optimizer.zero_grad()

        # 假设原始目标序列：
        # trg = [<sos>, I, like, NLP, <eos>]

        # 模型学习过程
        # 第 0 步（输入 <sos>） → 预测 "I"
        # 第 1 步（输入 <sos>, I） → 预测 "like"
        # 第 2 步（输入 <sos>, I, like） → 预测 "NLP"
        # 第 3 步（输入 <sos>, I, like, NLP） → 预测 "<eos>"
        # decoder 的输入总是目标序列的左移一位版本，所以是 trg[:, :-1]
        # decoder 的期望输出是目标序列本身去掉第一个 token，所以是 trg[:, 1:]

        # trg[:, :-1]：shifted trg（输入给 decoder）
        output = model(src, trg[:, :-1])

        # tensor.contiguous 是 PyTorch 张量的一个方法，用来返回内存连续（contiguous）的副本。
        # tensor.view 是 PyTorch 张量的一个方法，用来返回一个新形状的张量。
        output_reshape = output.contiguous().view(-1, output.shape[-1])

        # decoder 的期望输出是目标序列本身去掉第一个 token
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()

        # 梯度裁剪
        # 在反向传播算出梯度后，如果梯度过大，就把它「限制」在一个范围内,这样避免参数更新幅度过大导致训练不稳定或发散。
        # 例如：按范数裁剪（clip by norm），把所有梯度的 L2 范数限制到 max_norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        logging.info(f'step : {round((i / len(iterator)) * 100, 2)} %%, loss : {loss.item()}')

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    """
    评估模型
    Args:
        model: 模型
        iterator: 数据迭代器
        criterion: 损失函数
    """
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), saved_path + 'model-{0}.pt'.format(valid_loss))

        f = open( train_loss_path, 'w')
        f.write(str(train_losses))
        f.close()

        f = open(bleu_path, 'w')
        f.write(str(bleus))
        f.close()

        f = open(test_loss_path, 'w')
        f.write(str(test_losses))
        f.close()

        logging.info(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        logging.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
        logging.info(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):.3f}')
        logging.info(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=num_epochs, best_loss=inf)
# %%

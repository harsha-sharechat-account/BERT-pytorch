import torch
from dataset import *
from trainer.pretrain import BERTTrainer
from dataset import *
from torch.utils.data import DataLoader

model_path='/home/sriharshkamma/final/BERT-pytorch/output/bert.model.ep6'
vocab_path='/home/sriharshkamma/final/BERT-pytorch/data/vocab.small'
test_dataset='/home/sriharshkamma/final/BERT-pytorch/data/corpus_little.small'

seq_len=20
batch_size=64
num_workers=5
log_freq=10
lr=1e-3
adam_beta1=0.9
adam_beta2=0.999
adam_weight_decay=0.01
with_cuda=True
on_memory=True
cuda_devices=None

vocab = WordVocab.load_vocab(vocab_path)
print("Vocab Size: ", len(vocab))
eval_dataset = BERTDataset(test_dataset, vocab, seq_len=seq_len, on_memory=on_memory)
eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=num_workers)
bert = torch.load(model_path)
trainer = BERTTrainer(bert, len(vocab), train_dataloader=None, test_dataloader=eval_data_loader,
                          lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay,
                          with_cuda=with_cuda, cuda_devices=cuda_devices, log_freq=log_freq)

trainer.test(0)

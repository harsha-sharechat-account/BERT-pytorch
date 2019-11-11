import torch.nn as nn
import numpy as np
import torch

# class TokenEmbedding(nn.Embedding):
#     def __init__(self, vocab_size, embed_size=512):
#         super().__init__(vocab_size, embed_size, padding_idx=0)
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__()
        path='/home/sriharshkamma/final/BERT-pytorch/fv.npy'
        embedding_array=np.load(path)
        self.embedding_dim=embed_size
        self.embedding=nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_array))
        
    def forward(self,x):
        return self.embedding(x)
        

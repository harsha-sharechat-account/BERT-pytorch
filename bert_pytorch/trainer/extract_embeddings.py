from bert_pytorch.model import BERT
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

class BERTEXTRACTOR:
    

    def __init__(self, bert: BERT ,train_dataloader: DataLoader,bert_model_path, with_cuda: bool = True, cuda_devices=None):
        
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = torch.load(bert_model_path)

        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.data = train_dataloader
        
    def extract(self,epoch):
        str_code = "extract embeddings"
        data_iter = tqdm.tqdm(enumerate(self.data),desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),bar_format="{l_bar}{r_bar}")
       
        prev=None
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            x=self.bert.forward(data["bert_input"], data["segment_label"])
            if i>0:
                prev=torch.stack([prev,x],dim=0)
            else:
                prev=x
        torch.save(prev, '/home/sriharshkamma/final/BERT-pytorch/tensor-pytorch.pt') 
    
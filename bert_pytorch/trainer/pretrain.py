import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from bert_pytorch.model import BERTLM, BERT
from .optim_schedule import ScheduledOptim
import pickle
import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, restore_model_path=None):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")#torch.device("cuda:0" if cuda_condition else "cpu")
        
        # restore model from epoch 
        self.restore_model_path=restore_model_path
        
        # This BERT model will be saved every epoch
        self.bert = bert
        #import pdb;pdb.set_trace()
        print('self.device=',self.device)
        # Initialize the BERT Language Model, with BERT model
        if restore_model_path==None:
            self.model = BERTLM(bert, vocab_size).to(self.device)
        else:
            
            self.model=torch.load(self.restore_model_path)
            self.model.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            print('cuda devices=',cuda_devices)
            self.model = nn.DataParallel(self.model, device_ids=[0])

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param

        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq
        print("Total Params:", sum([p.nelement() for p in self.model.parameters()]),\
              'train params =',count_parameters(self.model))

    def train(self, epoch):
        return self.iteration(epoch, self.train_data)

    def test(self, epoch):
        return self.iteration(epoch, self.test_data, train=0)
      
    def extract(self,epoch):
        self.iteration(epoch,self.test_data,train=2)

    def iteration(self, epoch, data_loader, train=1):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        if train ==1:
            str_code = "train"
        if train ==0:
            str_code = "test"
        if train ==2:
            str_code = "extract embeddings"
            
        # Setting the tqdm progress bar 
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        
        if train==2:
            prev=None
            for i, data in data_iter:
                data = {key: value.to(self.device) for key, value in data.items()}
                x=self.bert.forward(data["bert_input"], data["segment_label"])
                if i>0:
                    prev=torch.stack([prev,x],dim=0)
                else:
                    prev=x
            torch.save(prev,'/home/sriharshkamma/final/BERT-pytorch/tensor-pytorch.pt') 
            
        
        if train ==1 or train ==0:
            for i, data in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in data.items()}

                # 1. forward the next_sentence_prediction and masked_lm model
                next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

                # 2-1. NLL(negative log likelihood) loss of is_next classification result
                next_loss = self.criterion(next_sent_output, data["is_next"])

                # 2-2. NLLLoss of predicting masked token word
                mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

                # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
                loss = next_loss + mask_loss

                # 3. backward and optimization only in train
                if train:
                    self.optim_schedule.zero_grad()
                    loss.backward()
                    self.optim_schedule.step_and_update_lr()

                # next sentence prediction accuracy
                correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
                avg_loss += loss.item()
                total_correct += correct
                total_element += data["is_next"].nelement()

                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "avg_acc": total_correct / total_element * 100,
                    "loss": loss.item()
                }

                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))

            print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
                  total_correct * 100.0 / total_element)
            return avg_loss / len(data_iter)
            
      

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "complete_model.ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        
        output_path_bert = file_path + "bert_only.ep%d" % epoch
        torch.save(self.bert.cpu(), output_path_bert)
        self.bert.to(self.device)
        
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

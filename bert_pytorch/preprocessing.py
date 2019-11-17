import sentencepiece as spm
import numpy as np
import os
import pickle
import fasttext
import argparse
def train_sentencepiece_model(path_to_text_data,vocab_size,model_name='model_sentencepiece'):
    
    spm.SentencePieceTrainer.Train('--input='+str(path_to_text_data)+' --model_prefix='+str(model_name)+' --vocab_size='+str(vocab_size))
#     spm.SentencePieceTrainer.Train('--input=/home/sriharshkamma/final/BERT-pytorch/data/version1_test_corpus.small --model_prefix='+str(model_name)+' --vocab_size=10000')
    
def create_vocab_file(sentence_piece_model_path,directory_path,vocab_file_name):
    sp=spm.SentencePieceProcessor()
    sp.load(sentence_piece_model_path)
    vocab_sentence_piece=[]
    for i in range(sp.get_piece_size()):
        vocab_sentence_piece.append(sp.id_to_piece(i))
    print(directory_path+'/temp.small')
    with open(directory_path+'/temp.small','w') as f:
        for i in vocab_sentence_piece:
            f.write(i+'\n')
    print('python dataset/vocab.py -c '+directory_path+'/temp.small'+' -o '+directory_path+'/'+vocab_file_name+'.small')
    os.system('bert-vocab -c '+directory_path+'/temp.small'+' -o '+directory_path+'/'+vocab_file_name+'.small')
    os.system('rm -r '+directory_path+'/temp.small')
      
    
def prepare_data(i):
    if len(i.strip())!=0 and len(i.split())>2:
        te=''
        m=len(i.split())
        c=0
        for j in i.split():
            if c==(m//2) -1 :
                te+=j+' \t '
            else:
                te+=j+' '
            c+=1
        return te.strip().replace('.','')
    else:
        return ''

def change_data_to_input_format(file_path_input,file_path_output,sentence_piece_model_path):
    with open(file_path_input,'r') as f:
        data=f.read()
    data=data.split('\n')[:-1]
    sp=spm.SentencePieceProcessor()
    sp.load(sentence_piece_model_path)
    req_data=[]
    for i in data:   
        changed_data=prepare_data(' '.join(sp.EncodeAsPieces(i)))
        if changed_data!='':
            req_data.append(changed_data)
    with open(file_path_output,'w') as f:
        for i in req_data:
            f.write(i+'\n')
            
def load_vocab(vocab_path: str):
    with open(vocab_path, "rb") as f:
        return pickle.load(f)
    
def create_fasttext_embeddings(vocab_file_path,fasttext_model_bin_path,path_to_save_fasttext_embeddings):
    vocab_data=load_vocab(vocab_file_path)
    model=fasttext.load_model(fasttext_model_bin_path)
    arr=[]
    for i in vocab_data.stoi:
        arr.append(model.get_sentence_vector(i))
    arr=np.array(arr)
    np.save(path_to_save_fasttext_embeddings,arr)
    
def train_fasttext_model(input_text_file,save_model_path):  
    model=fasttext.train_unsupervised(input_text_file,model='skipgram',wordNgrams=3,verbose=1,dim=100,epoch=100)
    model.save_model(save_model_path)
def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-path_to_text_data", "--path_to_text_data", required=True, type=str, help="path to text data")
    parser.add_argument("-vocab_size", "--vocab_size", required=True, type=int,help="vocab_size")
    parser.add_argument("-path_to_modified_text_data", "--path_to_modified_text_data", type=str, help="path to modified data")
    parser.add_argument("-path_to_fasttext_model", "--path_to_fasttext_model", type=str, help="path to fasttext model")
    
    args = parser.parse_args()
    
    train_sentencepiece_model(args.path_to_text_data,args.vocab_size)
    create_vocab_file(os.getcwd()+'/model_sentencepiece.model',os.getcwd(),'vocabfile')
    print('vocab file created')
    change_data_to_input_format(args.path_to_text_data,args.path_to_modified_text_data,os.getcwd()+'/model_sentencepiece.model')
    print('imput format created')
    train_fasttext_model(args.path_to_modified_text_data,args.path_to_fasttext_model)
    print('fasttext model created')
    create_fasttext_embeddings(os.getcwd()+'/vocabfile.small',args.path_to_fasttext_model,'./fasttext_vectors.npy')
    #'/home/sriharshkamma/model_filename_big.bin'
   
if __name__ == "__main__":
    train()    
#python preprocessing.py -path_to_text_data /home/sriharshkamma/final/BERT-pytorch/data/version1_test_corpus.small -vocab_size 100 -path_to_modified_text_data ./modified_text_data.small -path_to_fasttext_model ./fasttext.bin
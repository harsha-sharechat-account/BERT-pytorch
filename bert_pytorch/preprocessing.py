import sentencepiece as spm
import numpy as np
import os
import pickle
import fasttext

def train_sentencepiece_model(path_to_text_data,vocab_size,model_name='model_sentencepiece'):
    spm.SentencePieceTrainer.Train('--input='+str(path_to_text_data)+' --model_prefix='+str(model_name)+' --vocab_size='+str(vocab_size))
    
def create_vocab_file(sentence_piece_model_path,directory_path,vocab_file_name):
    sp=spm.SentencePieceProcessor()
    sp.load(sentence_piece_model_path)
    vocab_sentence_piece=[]
    for i in range(sp.get_piece_size()):
        vocab_sentence_piece.append(i)
    with open(directory_path+'/temp.small','w') as f:
        for i in vocab_sentence_piece:
            f.write(i+'\n')
    os.system('python dataset/vocab.py -c '+directory_path+'/temp.small'+' -o '+directory_path+'/'+vocab_file_name+'.small')
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
        ne.append(te.strip().replace('.',''))
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
        changed_data=prepare_data(' '.join(sp.EncodeAsPieces(data[i])))
        if changed_data!='':
            req_data.append(changed_data)
    with open(file_path_output,'w') as f:
        for i in req_data:
            f.write(i+'\n')
            
def load_vocab(vocab_path: str):
    with open(vocab_path, "rb") as f:
        return pickle.load(f)
    
def create_fasttext_embeddings(vocab_file_path,fasttext_model_bin_path,path_to_save_fasttext_embeddings):
    vocab_data=load_vocab(path)
    model=fasttext.load_model(fasttext_model_bin_path)
    arr=[]
    for i in vocab_data.stoi:
        arr.append(model.get_sentence_vector(i))
    arr=np.array(arr)
    np.save(path_to_save_fasttext_embeddings,arr)
    

    
    
    
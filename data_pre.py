# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:40:34 2021

@author: LYZ
"""
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer
import numpy as np
import json
import nltk
import util.config as config

# cpu or cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 68
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


# read original data
def data_reader(args):
    with open(args.data_load_path) as f:
        data = json.load(f)
    text_list = list(data.keys())
    label_list, pos_topic_list = [], []
    value_list = list(data.values())
    if "\t" in value_list[0]:
        for v in value_list:
            v_list = v.split("\t")
            values = [int(n) for n in v_list]
            label_list.append(values[0])
            # pos_topic = torch.from_numpy(np.array(values[1:]))
            pos_topic_list.append(values[1:])    
    else:
        for value in value_list:
            label_list.append(value)
            pos_topic_list.append(-1)
            
    pos_topic_list = torch.from_numpy(np.array(pos_topic_list))
         
    return text_list, label_list, pos_topic_list

# create tag tensor
def tag(tag_list, word_list, args):
    # 0:pad; 1:framing tokens; 2:topic tokens; 3:[CLS], [SEP]
    # topic_words = ["NN", "NNP", "NNS", "NNPS", "FW", "PDT", "POS", "PRP", "PRP"]
    topic_words = ["NN", "NNP", "NNS", "NNPS"]
    text_tag = torch.zeros(1, args.sentence_max_length)
    text_length = torch.zeros(1, 2)
    
    if len(tag_list) > args.sentence_max_length-2:
        tag_list = tag_list[:args.sentence_max_length-2]
        word_list = word_list[:args.sentence_max_length-2]
        
    # 1: framing tokens, 2: topic tokens
    for i in range(len(tag_list)):
        if tag_list[i] in topic_words:
            text_tag[0][i+1] = 2
        else:
            text_tag[0][i+1] = 1
    
    # set [SEP] and [CLS] as 3
    text_tag[0][0], text_tag[0][len(tag_list)+1] = 3, 3
    sep_index = -2
    for i,w in enumerate(word_list):
        if w == "[SEP]":
            sep_index = i
            break
    if sep_index > 0:
        text_tag[0][sep_index+1] = 3
    
    # length of 2 sentences length
    # text_length[.][0]: the first sentence length
    # text_length[.][1]: the second sentence length 
    # 2th sentence length is -2 when only 1 sentence: text_length[.][0]=-2
    text_length[0][0] = sep_index
    text_length[0][1] = len(tag_list)
           
    return text_tag, text_length
        
    

# sentences --> word ids
def data_encode_tag(args, text_list, label_list):
    '''
    Returns
    -------
    text_id_list : tensor [sentences number, sentences dimension].
    label_list : tensor [sentences number].
    text_tag_list : tensor [sentences number, sentences dimension].
    '''
    
    text_id_list, text_tag_list, text_length_list = [], [], []
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_model,
                                              do_lower_case=True)
    for text in text_list:
        # text tag
        text = text.strip()
        word_list = tokenizer.tokenize(text)
        tag_list = [t[1] for t in nltk.pos_tag(word_list)]
        text_tag, text_length = tag(tag_list, word_list, args)
        text_tag_list.append(text_tag)
        text_length_list.append(text_length)
        # text encode
        # sentence max length: 100 default
        text_id = tokenizer.encode(text, 
                                   return_tensors = 'pt', # return type: pt (pytorch tensor)
                                   add_special_tokens=True, # add_special_tokens: add CLS,SEP
                                   max_length=args.sentence_max_length,
                                   pad_to_max_length = True, truncation=True
                                   ).to(device)
        text_id_list.append(text_id)
        
    text_tag_list = torch.cat(text_tag_list, dim=0).to(device)
    text_id_list = torch.cat(text_id_list, dim=0).to(device)
    text_length_list = torch.cat(text_length_list, dim=0).to(device)
    label_list = torch.tensor(label_list).to(device)
    # print(text_id_list.shape, label_list.shape)
    
    return text_id_list, label_list, text_tag_list, text_length_list

# create train, val, test data loader
def data_loader(args):
    text_list, label_list, pos_topic_list = data_reader(args)
    text_id_list, label_list, text_tag_list, text_length_list = data_encode_tag(args, text_list, label_list)
    
    # Split data into train, val, test
    dataset = TensorDataset(text_id_list, label_list, text_tag_list, 
                            text_length_list, pos_topic_list)
    train_size = int(args.train_size * len(dataset))
    val_size = int(args.val_size * len(dataset))
    test_size = len(dataset) - train_size - val_size   
    
    if args.data_split_type == "random":
        train_dataset, val_dataset, test_dataset = random_split(dataset,
                                                [train_size, val_size, test_size])
    else:
        train_dataset = TensorDataset(text_id_list[:train_size,:],
            label_list[:train_size], text_tag_list[:train_size,:], 
            text_length_list[:train_size,:], pos_topic_list[:train_size,:])
        val_dataset = TensorDataset(text_id_list[train_size:train_size+val_size,:],
            label_list[train_size:train_size+val_size],
            text_tag_list[train_size:train_size+val_size,:], 
            text_length_list[train_size:train_size+val_size,:],
            pos_topic_list[train_size:train_size+val_size,:])
        test_dataset = TensorDataset(text_id_list[train_size+val_size:,:],
            label_list[train_size+val_size:], text_tag_list[train_size+val_size:,:], 
            text_length_list[train_size+val_size:,:],
            pos_topic_list[train_size+val_size:,:])
        
    # Create train, val, test dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    # tokenize()
    parser = config.get_config()
    args = parser.parse_args()
    data_loader(args)
    
    
    
    
    
    
    
    
    
    



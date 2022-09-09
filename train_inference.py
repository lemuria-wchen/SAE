# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:40:34 2021

@author: LYZ
"""
import random
import torch
import torch.nn as nn
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import util.config as config
import data_pre
from tqdm import tqdm
import util.measure as measure
import util.utils as utils
import model.BERT_arg as BERT_arg
import model.BERT_tokentype as BERT_tokentype
import time
import json

# cpu or cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seed = 68
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True       


def train(args, dataloader, model, optimizer, scheduler, 
          attention_mask_list, token_type_ids_list=None):
    model.train()
    preds_list, labels_list = [], []
    total_loss = 0  
    index = 0
    
    for data in tqdm(dataloader):
        attention_mask = attention_mask_list[index]
        if token_type_ids_list:
            token_type_ids = token_type_ids_list[index]
        else:
            token_type_ids = None
        model.zero_grad()
        
        texts = data[0].to(device)
        labels = data[1].to(device)
        pos_topic = data[4].to(device)
        
        loss, logits = model(texts, pos_topic=pos_topic, 
                             token_type_ids=token_type_ids, 
                             attention_mask=attention_mask, labels=labels)
        
        # print("======Train", loss.shape, logits.shape)
        if torch.cuda.device_count() > 1:
            loss = torch.sum(loss) / torch.cuda.device_count()
        
        preds = torch.argmax(logits, dim=1)
        
        total_loss += loss.item()        
        preds_list += preds.cpu().detach().numpy().tolist()
        labels_list += labels.cpu().detach().numpy().tolist()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step() 
        scheduler.step()
        
        index += 1
    
    loss = total_loss / len(dataloader)
    acc, p, r, f1 = measure.measures(preds_list, labels_list)
    
    return loss, acc, p, r, f1

def evaluate(args, dataloader, model, log_file, 
             attention_mask_list, token_type_ids_list=None):
    model.eval()
    preds_list, labels_list = [], []
    total_loss = 0
    index = 0
    
    for data in tqdm(dataloader):
        with torch.no_grad():
            attention_mask = attention_mask_list[index]
            if token_type_ids_list:
                token_type_ids = token_type_ids_list[index]
            else:
                token_type_ids = None
            texts = data[0].to(device)
            labels = data[1].to(device)
            
            if args.transfer_type == "2-5":
                for i in range(len(labels)):
                    if labels[i] == 3: labels[i] = 1
                    else: labels[i] = 0
            
            pos_topic = data[4].to(device)
        
            loss, logits = model(texts, pos_topic=pos_topic, 
                                 token_type_ids=token_type_ids, 
                                 attention_mask=attention_mask, labels=labels)
        
            if torch.cuda.device_count() > 1:
                loss = torch.sum(loss) / torch.cuda.device_count()
            
            preds = torch.argmax(logits, dim=1)
        
            total_loss += loss.item()        
            preds_list += preds.cpu().detach().numpy().tolist()
            labels_list += labels.cpu().detach().numpy().tolist()
            
            index += 1
    
    loss = total_loss / len(dataloader)
    
    # label_path = "./preds/BioMedical_labels.json"
    # label_file = open(label_path, "w")
    # label_file.write(json.dumps(labels_list, indent=4, ensure_ascii=False))
    # label_file.close()
    
    # write predict labels
    preds_path = "./preds/" + args.preds_name + ".json"
    preds_file = open(preds_path, "w")
    preds_file.write(json.dumps(preds_list, indent=4, ensure_ascii=False))
    preds_file.close()
    sen = "** predict labels --> {}".format(preds_path)
    print(sen)
    log_file.write(sen + "\n")
    
    
    acc, p, r, f1 = measure.transfer_measures(args.transfer_type, 
                                              preds_list, labels_list)
    
    return loss, preds_list, acc, p, r, f1
    

def main():
    
    # get config parameters
    parser = config.get_config()
    args = parser.parse_args()
    config.print_config(args)
    
    # get bert config
    bert_config = config.get_bert_config(args)
    
    # write log info
    log_file = open("./log/" + args.log_name + ".txt", "w")
    
    # load data
    print("** Loading Data ...")
    start_time = time.time()
    train_dataloader, val_dataloader, test_dataloader = data_pre.data_loader(args)
    end_time = time.time()
    mins, secs = divmod(end_time - start_time, 60)
    sen = "** Load Data Done! Time: {}m {:.2f}s".format(mins, secs)
    print(sen)
    log_file.write(sen + "\n")
    
    # create model
    if args.type == "tokentype":
        model = BERT_tokentype.BERT_tokentype(args, bert_config)
    else:
        model = BERT_arg.BERT_arg(args, bert_config)
        
    # muti GPUs
    if torch.cuda.device_count() > 1:
        sen = "** GPU: Use {} GPUs".format(torch.cuda.device_count())
        print(sen)
        log_file.write(sen + "\n")
        model = nn.DataParallel(model)
    else:
        sen = "** CPU: Use CPUs"
        print(sen)
        log_file.write(sen + "\n")
    
    model.to(device)   
    
    
    # load model and predict
    model.load_state_dict(torch.load("./output/" + args.model_save_name + ".pt"))
    
    # get test attention mask
    print("** Getting Test Attention Mask ...")
    start_time = time.time()
    attention_mask_list_test = utils.get_attention_mask(test_dataloader, args.arg_encoding)
    end_time = time.time()
    mins, secs = divmod(end_time - start_time, 60)
    sen = "** Getting Test Attention Mask Done! Time: {}m {:.2f}s".format(mins, secs)
    print(sen)
    log_file.write(sen + "\n")
    
    # get test token type ids
    print("** Getting Test Token Type Ids ...")
    start_time = time.time()
    token_type_ids_list_test = None
    if args.type == "tokentype":
        token_type_ids_list_test = utils.get_token_type_ids(test_dataloader)
    end_time = time.time()
    mins, secs = divmod(end_time - start_time, 60)
    sen = "** Getting Test Token Type Ids Done! Time: {}m {:.2f}s".format(mins, secs)
    print(sen)
    log_file.write(sen + "\n")
    
    # test
    print("** Testing ...")
    test_loss, preds_list, test_acc, test_p, test_r, test_f1 = evaluate(args,
                test_dataloader, model, log_file,
                attention_mask_list_test,
                token_type_ids_list_test)
    
    # print report
    measure.transfer_report_log("Test", test_loss, test_acc, 
                                test_p, test_r, test_f1, log_file, args)
    # close log file
    log_file.close()
    
    
    

if __name__ == "__main__":
    main()
    



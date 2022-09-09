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

# cpu or cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seed = 68
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True       


def train(dataloader, model, optimizer, scheduler, 
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

def evaluate(dataloader, model, attention_mask_list, token_type_ids_list=None):
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
    acc, p, r, f1 = measure.measures(preds_list, labels_list)
    
    return loss, acc, p, r, f1
    

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
    
    # experiment setting
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=total_steps)   
    best_measure = float('-inf')
    best_epoch = -1
    
    # get train attention mask
    print("** Getting Train Attention Mask...")
    start_time = time.time()
    attention_mask_list_train = utils.get_attention_mask(train_dataloader, args.arg_encoding)
    end_time = time.time()
    mins, secs = divmod(end_time - start_time, 60)
    sen = "** Getting Train Attention Mask Done! Time: {}m {:.2f}s".format(mins, secs)
    print(sen)
    log_file.write(sen + "\n")
   
    # get dev attention mask
    print("** Getting Dev Attention Mask ...")
    start_time = time.time()
    attention_mask_list_dev = utils.get_attention_mask(val_dataloader, args.arg_encoding)
    end_time = time.time()
    mins, secs = divmod(end_time - start_time, 60)
    sen = "** Getting Dev Attention Mask Done! Time: {}m {:.2f}s".format(mins, secs)
    print(sen)
    log_file.write(sen + "\n")
    
    # get token type ids
    print("** Getting Token Type Ids ...")
    start_time = time.time()
    token_type_ids_list_train, token_type_ids_list_dev = None, None
    if args.type == "tokentype":
        token_type_ids_list_train = utils.get_token_type_ids(train_dataloader)
        token_type_ids_list_dev = utils.get_token_type_ids(val_dataloader)
    end_time = time.time()
    mins, secs = divmod(end_time - start_time, 60)
    sen = "** Getting Token Type Ids Done! Time: {}m {:.2f}s".format(mins, secs)
    print(sen)
    log_file.write(sen + "\n")
    
      
    for epoch in range(args.epochs):
        print("="*13 + "epoch " + str(epoch+1) + "="*13)
        
        start_time = time.time()
        
        # train model
        print("** Epoch-{} Training ...".format(epoch+1))
        train_loss, train_acc, train_p, train_r, train_f1 = train(train_dataloader,
                        model, optimizer, scheduler, 
                        attention_mask_list_train,
                        token_type_ids_list_train)
        
        # dev
        print("** Epoch-{} Evaluating ...".format(epoch+1))
        dev_loss, dev_acc, dev_p, dev_r, dev_f1 = evaluate(val_dataloader, model, 
                        attention_mask_list_dev,
                        token_type_ids_list_dev)
        
        end_time = time.time()
        mins, secs = divmod(end_time - start_time, 60)
        
        # save model when better measure 
        if dev_f1["avg"] > best_measure:
            best_measure = dev_f1["avg"]
            best_epoch = epoch
            torch.save(model.state_dict(),
                       "./output/" + args.model_save_name + ".pt")
        
        # print train and dev report and write log file
        sen = "Epoch: {:02} | Epoch Time: {}m {:.2f}s".format(epoch+1, mins, secs)
        log_file.write(sen + "\n")
        print(sen)
        
        measure.report_log("Train", train_loss, train_acc, 
                           train_p, train_r, train_f1, log_file, args)
        measure.report_log("Dev", dev_loss, dev_acc, 
                           dev_p, dev_r, dev_f1, log_file, args)
    
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
    test_loss, test_acc, test_p, test_r, test_f1 = evaluate(test_dataloader, model, 
                        attention_mask_list_test,
                        token_type_ids_list_test)
    # print Test report and write log file
    measure.report_log("Test", test_loss, test_acc, 
                       test_p, test_r, test_f1, log_file, args)
    sen = "Best epoch is epoch {}".format(best_epoch+1)
    print(sen)
    log_file.write(sen + "\n")
    # close log file
    log_file.close()

if __name__ == "__main__":
    main()
    



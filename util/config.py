# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:40:34 2021

@author: LYZ
"""
from transformers import BertConfig
import argparse
import os

#获取当前文件夹路径
basedir = os.path.abspath(os.path.dirname(__file__))

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_load_path", default="./data/sentence_type/MyCorpus_type.json",
                        type=str, help="Path of data")
    parser.add_argument("--cross_data_load_path", default="./data/sentence_type/ScientificPublications_type.json",
                        type=str, help="Path of cross data")
    parser.add_argument("--label_dim", default=5,
                        type=int, help="Number of label")
    parser.add_argument("--transfer_label_dim", default=2,
                        type=int, help="Number of transfer label")
    parser.add_argument("--batch_size", default=32,
                        type=int, help="batch size")
    parser.add_argument("--epochs", default=6,
                        type=int, help="number of epochs")
    parser.add_argument("--lr", default=2e-5,
                        type=float, help="learning rate")
    parser.add_argument("--dropout", default=0.1,
                        type=float, help="dropout probability")
    # parser.add_argument("--bert_pretrain_model", default="D:/BERT/PretrainedModel/pytorch/bert-base-uncased",
    #                     type=str, help="pretrained model path")
    parser.add_argument("--bert_pretrain_model", default="./pretrained_model/bert-base-uncased",
                        type=str, help="pretrained model path")
    parser.add_argument("--arg_encoding", default="True",
                        type=str, help="whether use argumentative encoding")   
    parser.add_argument("--arg_encoding_type", default="max",
                        type=str, help="argumentative encoding concate type: max/avg/concate")
    parser.add_argument("--model_save_name", default="model",
                        type=str, help="name of saved model")
    parser.add_argument("--sentence_max_length", default=100,
                        type=int, help="name of saved model")
    parser.add_argument("--data_split_type", default="order",
                        type=str, help="type of splitting data: random / order")
    parser.add_argument("--lstm", default="True",
                        type=str, help="whether use lstm layer")
    parser.add_argument("--lstm_hidden_dim", default=200,
                        type=int, help="hidden layer dim of LSTM")
    parser.add_argument("--log_name", default="log",
                        type=str, help="name of log name")
    parser.add_argument("--preds_name", default="log",
                        type=str, help="name of predict labels file name")
    parser.add_argument("--PTE", default="position",
                        type=str, help="PTE: position / topic / PTE / no")
    parser.add_argument("--type", default="argencoding",
                        type=str, help="how to add token type: argencoding or tokentype")
    parser.add_argument("--train_size", default=0.6,
                        type=float, help="proportion of training set")
    parser.add_argument("--val_size", default=0.2,
                        type=float, help="proportion of validation set")
    parser.add_argument("--transfer_type", default="5-2",
                        type=str, help="x1-x2, x1 class model transfer to x2 class dataset")
    
    

    return parser

def get_bert_config(args):
    bert_config = BertConfig.from_pretrained(args.bert_pretrain_model)
    bert_config.output_hidden_states = True
    bert_config.output_attentions = True
    # bert_config.add_cross_attention=True
    
    return bert_config
    
def print_config(args):
    print("="*16 + "v Config Parameters v" + "="*16)
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
    print("="*16 + "^ Config Parameters ^" + "="*16)

if __name__ == "__main__":
    parser = get_config()
    args = parser.parse_args()
    print_config(args)
    print(get_bert_config(args))
    







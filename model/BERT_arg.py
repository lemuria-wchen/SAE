# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:40:34 2021

@author: LYZ
"""
import torch
import torch.nn as nn
# pip install pytorch-transformers==1.0
from transformers import BertModel
from torch.nn import CrossEntropyLoss, MSELoss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BERT_arg(nn.Module):
    def __init__(self, args, bert_config):
        super(BERT_arg, self).__init__()
        self.label_dim = args.label_dim
        self.arg_encoding = args.arg_encoding        
        self.arg_encoding_type = args.arg_encoding_type
        self.lstm_use = args.lstm
        self.PTE = args.PTE

        self.bert = BertModel.from_pretrained(args.bert_pretrain_model, 
                                              config=bert_config).to(device)
        self.lstm = nn.LSTM(bert_config.hidden_size, args.lstm_hidden_dim, 
                              num_layers = 2, bidirectional = True,
                              batch_first=True, dropout = args.dropout).to(device)
        self.dropout = nn.Dropout(args.dropout).to(device)
        self.classifier_BERT = nn.Linear(bert_config.hidden_size, 
                                    self.label_dim).to(device)
        self.classifier_BERT_PTE = nn.Linear(bert_config.hidden_size+2, 
                                    self.label_dim).to(device)
        self.classifier_BERT_PorT = nn.Linear(bert_config.hidden_size+1, 
                                    self.label_dim).to(device)
        self.classifier_lstm_PTE = nn.Linear(args.lstm_hidden_dim*2 + 2, 
                                    args.lstm_hidden_dim).to(device)
        self.classifier_lstm_PorT = nn.Linear(args.lstm_hidden_dim*2 + 1, 
                                    args.lstm_hidden_dim).to(device)
        self.classifier_lstm = nn.Linear(args.lstm_hidden_dim*2, 
                                    args.lstm_hidden_dim).to(device)
        self.classifier_lstm_output = nn.Linear(args.lstm_hidden_dim, 
                                    self.label_dim).to(device)

    def forward(
        self,
        input_ids=None,
        pos_topic=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.label_dim - 1]`. If :obj:`config.label_dim == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.label_dim > 1` a classification loss is computed (Cross-Entropy).
        """

        # use argumentative encoding
        if self.arg_encoding in ["True", "true", "Yes", "yes", "Y", "y", 1]:
            # self: internal-attention, cross: external-attention; all: self-attention
            
            output_self = self.bert(
                input_ids,
                attention_mask=attention_mask["self"],
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
            
            output_cross = self.bert(
                input_ids,
                attention_mask=attention_mask["cross"],
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
            
            output_all = self.bert(
                input_ids,
                attention_mask=attention_mask["all"],
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
            
            # output_self[0]: [batch_size, seq_length, 768]
            # output_self[1]: [batch_size, 768]
            # use LSTM layer
            if self.lstm_use in ["True", "true", "Yes", "yes", "Y", "y", 1]:  
                # pooled_output: [3, batch_size, lstm_hidden_dim*2]
                
                output_self, (final_hidden_state, final_cell_state) = self.lstm(output_self[0])
                output_self = output_self[:,0,:] + output_self[:,-1,:]
                
                output_cross, (final_hidden_state, final_cell_state) = self.lstm(output_cross[0])
                output_cross = output_cross[:,0,:] + output_cross[:,-1,:]
                
                output_all, (final_hidden_state, final_cell_state) = self.lstm(output_all[0])
                output_all = output_all[:,0,:] + output_all[:,-1,:]               
                pooled_output = torch.cat([output_self.unsqueeze(0),
                                            output_cross.unsqueeze(0),
                                            output_all.unsqueeze(0)], dim=0)
                # pooled_output = torch.cat([output_self.unsqueeze(0),
                #                            output_all.unsqueeze(0)], dim=0)
            # no lstm layer
            else:                
                # pooled_output: [3, batch_size, 768]
                pooled_output = torch.cat([output_self[1].unsqueeze(0),
                                        output_cross[1].unsqueeze(0),
                                        output_all[1].unsqueeze(0)], dim=0)
                # pooled_output = torch.cat([output_self[1].unsqueeze(0),
                #                         output_all[1].unsqueeze(0)], dim=0)
            
            # max pooling
            if self.arg_encoding_type == "max":
                pooled_output = torch.max(pooled_output, dim=0)[0]
            # avg polling
            else:
                pooled_output = torch.sum(pooled_output, dim=0)/3
                
        # no argumentative encoding    
        else:
            output = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
            
            # print(output[0].shape)
            # print(type(output[0]))

            # output[0]: [batch_size, seq_length, 768]
            # output[1]: [batch_size, 768]
            # use LSTM layer
            if self.lstm_use in ["True", "true", "Yes", "yes", "Y", "y", 1]:
                # output: [batch_size, seq_length, 768]
                # pooled_output: [batch_size, lstm_hidden_dim*2]
                
                output, (final_hidden_state, final_cell_state) = self.lstm(output[0])
                pooled_output = output[:,0,:] + output[:,-1,:]
            # no LSTM layer
            else:
                # pooled_output: [batch_size, 768]
                pooled_output = output[1]
    
        
        pooled_output = self.dropout(pooled_output)
        # pos_topic = (torch.sum(pos_topic, dim=1)/pos_topic.shape[1]).unsqueeze(1)
        # pos_topic: [batch_size, 2]
        if self.PTE == "PTE":
            pooled_output =  torch.cat([pooled_output, pos_topic.unsqueeze(1)], dim=1)
            if self.lstm_use in ["True", "true", "Yes", "yes", "Y", "y", 1]:
                # [batch_size, lstm_hidden_dim*2+2] -> [batch_size, lstm_hidden_dim]
                pooled_output = self.classifier_lstm_PTE(pooled_output)
                # [batch_size, lstm_hidden_dim] -> [batch_size, label_dim]
                logits = self.classifier_lstm_output(pooled_output)
            else:
                # [batch_size, 768+2] -> [batch_size, label_dim]
                logits = self.classifier_BERT_PTE(pooled_output)
        elif self.PTE in ["position", "topic"]:
            if self.PTE == "position":
                pooled_output =  torch.cat([pooled_output, pos_topic[:,0].unsqueeze(1)], dim=1)
            else:
                pooled_output =  torch.cat([pooled_output, pos_topic[:,1].unsqueeze(1)], dim=1)
            if self.lstm_use in ["True", "true", "Yes", "yes", "Y", "y", 1]:
                # [batch_size, lstm_hidden_dim*2+1] -> [batch_size, lstm_hidden_dim]
                pooled_output = self.classifier_lstm_PorT(pooled_output)
                # [batch_size, lstm_hidden_dim] -> [batch_size, label_dim]
                logits = self.classifier_lstm_output(pooled_output)
            else:
                # [batch_size, 768+1] -> [batch_size, label_dim]
                logits = self.classifier_BERT_PorT(pooled_output)
        else:
            if self.lstm_use in ["True", "true", "Yes", "yes", "Y", "y", 1]:
                # [batch_size, lstm_hidden_dim*2] -> [batch_size, lstm_hidden_dim]
                pooled_output = self.classifier_lstm(pooled_output)
                # [batch_size, lstm_hidden_dim] -> [batch_size, label_dim]
                logits = self.classifier_lstm_output(pooled_output)
            else:
                # [batch_size, 768] -> [batch_size, label_dim]
                logits = self.classifier_BERT(pooled_output)

        loss = None
        if labels is not None:
            if self.label_dim == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.label_dim), labels.view(-1))

        # hidden_states = outputs.hidden_states
        # attentions = outputs.attentions
        
        return loss.to(device), logits.to(device)
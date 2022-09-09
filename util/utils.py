# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:40:34 2021

@author: LYZ
"""
import torch
from tqdm import tqdm
import os

# cpu or cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#获取当前文件夹路径
basedir = os.path.abspath(os.path.dirname(__file__))

def get_attention_mask(dataloader, arg_encoding):
    attention_mask_list = []
        
    for data in tqdm(dataloader):
        texts = data[0].to(device)
        tags = data[2].to(device)
        lengths = data[3].to(device)
        
        # print("111111", texts.shape, tags.shape, lengths.shape)
        
        attention_mask_dict = {}
        attention_mask = torch.rand(tags.shape[0], tags.shape[1], 
                                    tags.shape[1]).to(device)
        attention_mask_self = torch.rand(tags.shape[0], tags.shape[1], 
                                    tags.shape[1]).to(device)
        attention_mask_cross = torch.rand(tags.shape[0], tags.shape[1], 
                                    tags.shape[1]).to(device)
        attention_mask_all = torch.rand(tags.shape[0], tags.shape[1], 
                                    tags.shape[1]).to(device)
        for m in range(tags.shape[0]):
            for n in range(texts.shape[1]):
                if arg_encoding in ["True", "true", "Yes", "yes", "Y", "y", 1]:
                    if tags[m][n] in [0, 3]:
                        attention_mask_self[m,n,:] = (texts[m]>0).to(device)
                        attention_mask_cross[m,n,:] = (texts[m]>0).to(device)
                        attention_mask_all[m,n,:] = (texts[m]>0).to(device)
                    else:
                        # [CLS], [SEP] seted True
                        attention_mask_self[m,n,0] = True
                        if lengths[m][0].type(torch.uint8) > 0:
                            attention_mask_self[m,n,lengths[m][0].type(torch.uint8)+1] = True
                        attention_mask_self[m,n,lengths[m][1].type(torch.uint8)+1] = True
                        attention_mask_self[m,n,:] = (tags[m]==tags[m][n]).to(device)
                        
                        attention_mask_cross[m,n,0] = True
                        if lengths[m][0].type(torch.uint8) > 0:
                            attention_mask_self[m,n,lengths[m][0].type(torch.uint8)+1] = True
                        attention_mask_cross[m,n,lengths[m][1].type(torch.uint8)+1] = True
                        attention_mask_cross[m,n,:] = (tags[m]==(3-tags[m][n])).to(device)
                        
                        attention_mask_all[m,n,:] = (texts[m]>0).to(device)
                    
                else:
                    attention_mask[m,n,:] = (texts[m]>0).to(device)
        # print(attention_mask)
        attention_mask_dict["self"] = attention_mask_self
        attention_mask_dict["cross"] = attention_mask_cross
        attention_mask_dict["all"] = attention_mask_all
        
        if arg_encoding in ["True", "true", "Yes", "yes", "Y", "y", 1]:
            attention_mask_list.append(attention_mask_dict)
        else:
            attention_mask_list.append(attention_mask)
        
    return attention_mask_list



def get_token_type_ids(dataloader):
    token_type_ids_list = []
        
    for data in tqdm(dataloader):
        # tags: [batch_size, sentence_length]
        tags = data[2].to(device) 
        # token_type_ids = torch.zeros_like(tags).to(device)
        token_type_ids = torch.rand(tags.shape[0], tags.shape[1]).to(device)
        for m in range(tags.shape[0]):
            token_type_ids[m,:] = (tags[m]==1).to(device)
            '''
            for n in range(tags.shape[1]):
                if tags[m][n] in [0, 2, 3]:
                    token_type_ids[m,n] = 1
                else:
                    token_type_ids[m,n] = 0 
            '''
        token_type_ids = torch.IntTensor(token_type_ids.cpu().numpy()).to(device)
        token_type_ids_list.append(token_type_ids)
    
    # token_type_ids_list = torch.tensor([t for t in token_type_ids_list]).to(device)
        
    return token_type_ids_list

if __name__ == "__main__":
    pass
    







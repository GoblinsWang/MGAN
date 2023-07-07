import torch
import torch.nn as nn
import torch.nn.init
import numpy as np
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
from torchsummary import summary
# from pyramid_vig import DeepGCN, pvig_ti_224_gelu
# from GAT import GAT, GATopt
from transformers import BertModel, BertTokenizer
import random

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def process_caption(tokenizer, tokens, train=True):
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()

        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% randomly change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            # 10% randomly change token to random token
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    deleted_idx.append(len(output_tokens) - 1)
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    target = torch.Tensor(target).long()
    return target

# Language Model with BERT
class EncoderText(nn.Module):
    def __init__(self, embed_size, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, embed_size)
        # self.gpool = GPO(32, 32)

    def forward(self, x):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0)
        bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D
        # cap_len = lengths

        cap_emb = self.linear(bert_emb)

        return cap_emb
        # pooled_features, pool_weights = self.gpool(cap_emb, cap_len.to(cap_emb.device))

        # normalization in the joint embedding space
        # if not self.no_txtnorm:
        #     pooled_features = l2norm(pooled_features, dim=-1)

        # return pooled_features

# class TextEncoder(nn.Module):
#     def __init__(self, bert_path = None, ft_bert = False, bert_size = 768, embed_size = 512):
#         super(TextEncoder, self).__init__()
#         self.bert = BertModel.from_pretrained(bert_path)
#         self.tokenizer = get_tokenizer(bert_path)
#         self.max_seq_len = 32
#         if not ft_bert:
#             for param in self.bert.parameters():
#                 param.requires_grad = False
#             print('text-encoder-bert no grad')
#         else:
#             print('text-encoder-bert fine-tuning !')
#         self.embed_size = embed_size
#         self.fc = nn.Sequential(nn.Linear(bert_size, embed_size), nn.ReLU(), nn.Dropout(0.1))

#     def forward(self, captions):
#         captions = self.get_text_input(captions)
#         all_encoders, pooled = self.bert(captions.unsqueeze(0))
#         out = all_encoders[-1]
#         out = self.fc(out)
#         return out
    
#     def get_text_input(self, caption):
#         # print(caption)
#         caption_tokens = self.tokenizer.tokenize(caption)
#         caption_tokens = ['[CLS]'] + caption_tokens + ['[SEP]']
#         caption_ids = self.tokenizer.convert_tokens_to_ids(caption_tokens)
#         if len(caption_ids) >= self.max_seq_len:
#             caption_ids = caption_ids[:self.max_seq_len]
#         else:
#             caption_ids = caption_ids + [0] * (self.max_seq_len - len(caption_ids))
#         caption = torch.tensor(caption_ids)
#         return caption    
    
# def get_tokenizer(bert_path):
#     tokenizer = BertTokenizer(bert_path + 'vocab.txt')
#     return tokenizer

if __name__ == '__main__':
    # model = GAT(GATopt(20, 1))
    # inputs = torch.randn(16, 20, 7, 7)
    # print('inputs shape : ', inputs.shape)
    # outputs = model(inputs)
    # print('outputs shape : ', outputs.shape)
    
    # model = pvig_ti_224_gelu()
    # print(summary(model, (3, 224, 224), device="cpu"))
    # model.backbone[2].add_module('GAT', GAT(GATopt(96, 1)))
    # model.backbone[5].add_module('GAT', GAT(GATopt(240, 1)))
    # model.backbone[12].add_module('GAT', GAT(GATopt(384, 1)))
    # print(model)

    # inputs = torch.randn(16, 3, 224, 224)
    # print('inputs shape : ', inputs.shape)
    # low_feature, mid_feature, solo_feature = model(inputs)
    # print('low_feature shape : ', low_feature.shape)
    # print('mid_feature shape : ', mid_feature.shape)
    # print('solo_feature shape : ', solo_feature.shape)
    # vsa_model = VSA_Module()
    # outputs = vsa_model(low_feature, mid_feature, solo_feature)
    # print('outputs shape : ', outputs.shape)

    bert_path = "/home/wzm/crossmodal/uncased_L-12_H-768_A-12/"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = ["i'm hello world 22"]
    target = process_caption(tokenizer, inputs).unsqueeze(0)
    print("target shape: ", target.shape)
    model = EncoderText(512)
    outputs = model(target)
    print("outputs shape : ", outputs.shape)
    print("outputs : ", outputs)
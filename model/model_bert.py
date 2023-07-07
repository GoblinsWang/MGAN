import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from collections import OrderedDict
from .utils import *
import copy
import ast
#from .mca import SA,SGA
from .pyramid_vig import DeepGCN, pvig_ti_224_gelu
from .GAT import GAT, GATopt, ATT
# from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertModel

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

# cross attention
class CrossAttention(nn.Module):

    def __init__(self, opt={}):
        super(CrossAttention, self).__init__()

        self.att_type = opt['cross_attention']['att_type']
        dim = opt['embed']['embed_dim']

        if self.att_type == "soft_att":
            self.cross_attention = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
        elif self.att_type == "fusion_att":
            self.cross_attention_fc1 = nn.Sequential(
                nn.Linear(2*dim, dim),
                nn.Sigmoid()
            )
            self.cross_attention_fc2 = nn.Sequential(
                nn.Linear(2*dim, dim),
            )
            self.cross_attention = lambda x:self.cross_attention_fc1(x)*self.cross_attention_fc2(x)

        elif self.att_type == "similarity_att":
            self.fc_visual = nn.Sequential(
                nn.Linear(dim, dim),
            )
            self.fc_text = nn.Sequential(
                nn.Linear(dim, dim),
            )
        else:
            raise Exception

    def forward(self, visual, text):
        batch_v = visual.shape[0]
        batch_t = text.shape[0]

        if self.att_type == "soft_att":
            visual_gate = self.cross_attention(visual)

            # mm
            visual_gate = visual_gate.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)

            return visual_gate*text

        elif self.att_type == "fusion_att":
            visual = visual.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)

            fusion_vec = torch.cat([visual,text], dim=-1)

            return self.cross_attention(fusion_vec)
        elif self.att_type == "similarity_att":
            visual = self.fc_visual(visual)
            text = self.fc_text(text)

            visual = visual.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)

            sims = visual*text
            return F.sigmoid(sims) * text



class VSA_Module(nn.Module):
    def __init__(self, opt = {}):
        super(VSA_Module, self).__init__()

        # extract value
        channel_size = 256
        embed_dim = 512

        # sub sample
        self.LF_conv = nn.Conv2d(in_channels=240, out_channels=channel_size, kernel_size=2, stride=2) # 240
        self.HF_conv = nn.Conv2d(in_channels=384, out_channels=channel_size, kernel_size=1, stride=1) # 512

        # visual attention
        self.concat_attention = GAT(GATopt(512, 1))

        # solo attention
        # self.solo_attention = nn.Linear(in_features=256, out_features=embed_dim)

    def forward(self, lower_feature, higher_feature, solo_feature):

        # b x channel_size x 16 x 16
        lower_feature = self.LF_conv(lower_feature)
        higher_feature = self.HF_conv(higher_feature)

        # concat
        # [b x 512 x 7 x 7]
        concat_feature = torch.cat([lower_feature, higher_feature], dim=1)

        # residual
        # [b x 512 x 7 x 7]
        concat_feature = higher_feature.mean(dim=1,keepdim=True).expand_as(concat_feature) + concat_feature

        # attention
        # [b x 512 x 7 x 7]
        attent_feature = self.concat_attention(concat_feature)

        # [b x 512 x 1 x 1]
        attent_feature = F.adaptive_avg_pool2d(attent_feature, 1)

        # [b x 512]
        attent_feature = attent_feature.squeeze(-1).squeeze(-1)

        # solo attention
        solo_att = torch.sigmoid(attent_feature)
        solo_feature = solo_feature * solo_att

        return l2norm(solo_feature, -1)

# class EncoderText(nn.Module):

#     def __init__(self, vocab_size, word_dim = 300, embed_size = 512, num_layers = 1,
#                  use_bi_gru=False, no_txtnorm=False):
#         super(EncoderText, self).__init__()
#         self.embed_size = embed_size
#         self.no_txtnorm = no_txtnorm

#         # word embedding
#         self.embed = nn.Embedding(vocab_size + 1, word_dim) # 编号从1开始，所以需要加1

#         # caption embedding
#         self.use_bi_gru = use_bi_gru
#         self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

#         self.init_weights()

#     def init_weights(self):
#         self.embed.weight.data.uniform_(-0.1, 0.1)

#     def forward(self, x, lengths):
#         """Handles variable size captions
#         """
#         # Embed word ids to vectors
#         x = self.embed(x)
#         packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

#         # Forward propagate RNN
#         out, _ = self.rnn(packed)

#         # Reshape *final* output to (batch_size, hidden_size)
#         padded = pad_packed_sequence(out, batch_first=True)
#         cap_emb, cap_len = padded

#         if self.use_bi_gru:
#             cap_emb = (cap_emb[:,:,:cap_emb.size(2)/2] + cap_emb[:,:,cap_emb.size(2)/2:])/2

#         # normalization in the joint embedding space
#         if not self.no_txtnorm:
#             cap_emb = l2norm(cap_emb, dim=-1)

#         return cap_emb, cap_len


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=512, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.HF_conv = nn.Conv2d(in_channels=384, out_channels=channels, kernel_size=1, stride=1) # 512

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.GELU()

        self.weight = nn.Parameter(torch.tensor(0.01)) # init 0.5

    def forward(self, vis_feature, text_feature):
        vis_feature = self.HF_conv(vis_feature)
        # print("vis_feature shape", vis_feature.shape)
        inter_feature = text_feature.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 7, 7)
        # print("inter_feature shape", inter_feature.shape)
        add_feature = vis_feature + inter_feature
        local_att_feature = self.local_att(add_feature)
        gobal_att_feature = self.global_att(add_feature)
        gl_feature = local_att_feature + gobal_att_feature
        gl_scores = self.sigmoid(gl_feature)

        attention_feature = 2 * vis_feature * gl_scores + 2 * inter_feature * (1 - gl_scores)
        attention_feature = self.relu(F.adaptive_avg_pool2d(attention_feature, 1).squeeze(-1).squeeze(-1))
        final_feature = self.weight * attention_feature + (1 - self.weight) * text_feature
        return final_feature

def func_attention(query, context, smooth):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)
    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    attn = l2norm(attn, 2)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext

def xattn_score_i2t(images, captions):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    # print("images size:", images.shape)
    # print("captions size:", captions.shape)

    l_features = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        cap_i = captions[i].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        # print("images size:", images.shape)
        # print("cap_i_expand size:", cap_i_expand.shape)
        weiContext = func_attention(cap_i_expand, images, smooth=9)
        weiContext = F.adaptive_avg_pool2d(weiContext.transpose(0, 2).unsqueeze(0), 1)
        weiContext = weiContext.squeeze(-1).squeeze(-1)
        # print("weiContext shape :", weiContext.shape)
        l_features.append(weiContext)
    text_feature = torch.cat(l_features, dim=0)
    # print("text_feature shape :", text_feature.shape)
    return text_feature

class TextEncoder(nn.Module):
    def __init__(self, bert_path = None, ft_bert = False, bert_size = 768, embed_size = 512):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        if not ft_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print('text-encoder-bert no grad')
        else:
            print('text-encoder-bert fine-tuning !')
        self.embed_size = embed_size
        self.fc = nn.Sequential(nn.Linear(bert_size, embed_size), nn.ReLU(), nn.Dropout(0.1))

    def forward(self, captions):
        all_encoders, pooled = self.bert(captions)
        # out = pooled
        out = all_encoders[-1]
        out = self.fc(out)
        return out

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

class BaseModel(nn.Module):
    def __init__(self, bert_path, vocab_words, opt={}):
        super(BaseModel, self).__init__()

        # img feature
        # self.extract_feature = ExtractFeature(opt = opt)
        self.extract_feature = pvig_ti_224_gelu()
        self.HF_conv = nn.Conv2d(in_channels=384, out_channels=512, kernel_size=1, stride=1) # 512
        # vsa feature
        self.mvsa =VSA_Module(opt = opt)

        # text feature
        # text feature
        # self.text_feature = Skipthoughts_Embedding_Module(
        #     vocab= vocab_words,
        #     opt = opt
        # )
        self.text_feature = EncoderText(512)
        self.gat_cap = ATT(GATopt(512, 1))

        self.Eiters = 0

    def forward(self, img, captions):

        # extract features
        lower_feature, higher_feature, solo_feature = self.extract_feature(img)

        # mvsa featrues
        # global_feature = self.mvsa(lower_feature, higher_feature, solo_feature)
        visual_feature = solo_feature

        # extract local feature
        # local_feature = self.local_feature(input_local_adj, input_local_rep)
        
        # text features
        text_feature = self.text_feature(captions)
        text_feature = l2norm(torch.mean(text_feature, dim=1), dim = -1)
        # print("text_feature shape :", text_feature.shape)
        # cap_feature, cap_lengths= self.text_feature(captions, lengths)

        # cross attention
        # node_feature = self.HF_conv(higher_feature)
        # node_feature = node_feature.reshape(node_feature.shape[0], node_feature.shape[1], -1)
        # node_feature = node_feature.transpose(2, 1)
        # text_feature = xattn_score_i2t(node_feature, text_feature)
        # cap_gat = self.gat_cap(text_feature)
        # print("cap_gat shape :", cap_gat.shape)
        # text_feature = l2norm(torch.mean(cap_gat, dim=1), dim = -1)
        # print("text_feature shape :", text_feature.shape)
        sims = cosine_sim(visual_feature, text_feature)
        return sims

def factory(opt, bert_path, vocab_word, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    model = BaseModel(bert_path, vocab_word, opt)
    # print(model)
    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model




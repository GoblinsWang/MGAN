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
from model.utils import *
import copy
import ast
from .pyramid_vig import DeepGCN, pvig_ti_224_gelu
from .GAT import GAT, GATopt, ATT

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

# cross attention for MGAN
class TFEM(nn.Module):

    def __init__(self, opt={}):
        super(TFEM, self).__init__()
        self.att_type = opt['cross_attention']['att_type']
        dim = opt['embed']['embed_dim']

        channel_size = 512

        self.visual_conv = nn.Conv2d(in_channels=384, out_channels=channel_size, kernel_size=1, stride=1)
        # visual attention
        self.visual_attention = GAT(GATopt(512, 1))

        # self.bn_out = nn.BatchNorm1d(512)
        # self.dropout_out = nn.Dropout(0.5)

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
        visual_feature = self.visual_conv(visual)
        visual_feature = self.visual_attention(visual_feature)

        # [b x 512 x 1 x 1]
        visual_feature = F.adaptive_avg_pool2d(visual_feature, 1)

        # [b x 512]
        visual_feature = visual_feature.squeeze(-1).squeeze(-1)

        if self.att_type == "soft_att":
            visual_gate = self.cross_attention(visual_feature)

            # mm
            visual_gate = visual_gate.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)

            return visual_gate*text

        elif self.att_type == "fusion_att":
            visual = visual_feature.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)

            fusion_vec = torch.cat([visual,text], dim=-1)

            return self.cross_attention(fusion_vec)
        elif self.att_type == "similarity_att":
            visual = self.fc_visual(visual_feature)
            text = self.fc_text(text)

            visual = visual.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)

            sims = visual*text
            text_feature = F.sigmoid(sims) * text
            return text_feature
            # return l2norm(text_feature, -1)

class MGAM(nn.Module):
    def __init__(self, opt = {}):
        super(MGAM, self).__init__()
        # extract value
        channel_size = 256
        # sub sample
        self.LF_conv = nn.Conv2d(in_channels=240, out_channels=channel_size, kernel_size=2, stride=2) # 240
        self.HF_conv = nn.Conv2d(in_channels=384, out_channels=channel_size, kernel_size=1, stride=1) # 512
        # visual attention
        self.concat_attention = GAT(GATopt(512, 1))

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
        # return solo_feature
        return l2norm(solo_feature, -1)

        # dynamic fusion
        # global_feature = solo_feature
        # feature_gl = global_feature + attent_feature
        # dynamic_weight = self.dynamic_weight(feature_gl)
        # weight_global = dynamic_weight[:, 0].reshape(feature_gl.shape[0],-1).expand_as(global_feature)


        # weight_local = dynamic_weight[:, 1].reshape(feature_gl.shape[0],-1).expand_as(global_feature)

        # visual_feature = weight_global*global_feature + weight_local*attent_feature
        # visual_feature = self.fusion(solo_feature, attent_feature)
        # return l2norm(visual_feature, -1)

class BaseModel(nn.Module):
    def __init__(self, opt={}, vocab_words=[], vocab_size = 0):
        super(BaseModel, self).__init__()

        # img feature
        # self.extract_feature = ExtractFeature(opt = opt)
        self.extract_feature = pvig_ti_224_gelu()
        self.HF_conv = nn.Conv2d(in_channels=384, out_channels=512, kernel_size=1, stride=1) # 512
        # vsa feature
        self.mgam =MGAM(opt = opt)
        # text feature
        self.text_feature = Skipthoughts_Embedding_Module(
            vocab= vocab_words,
            opt = opt
        )
        self.tfem = TFEM(opt = opt)
        self.Eiters = 0

    def forward(self, img, captions, text_lens=None):

        # extract features
        lower_feature, higher_feature, solo_feature = self.extract_feature(img)

        # mgam featrues
        visual_feature = self.mgam(lower_feature, higher_feature, solo_feature)
        # visual_feature = solo_feature


        # text features
        # print("captions shape :", captions.shape)
        text_feature = self.text_feature(captions)
        # cap_feature, cap_lengths= self.text_feature(captions, lengths)

        # TFEM
        dual_text = self.tfem(higher_feature, text_feature)
        # dual_text = self.tfem(solo_feature, text_feature)
        # Ft = text_feature
        # print("Ft size : ", Ft.shape)

        # sim dual path
        dual_visual = visual_feature.unsqueeze(dim=1).expand(-1, dual_text.shape[1], -1)
        # print("mgam_feature size : ", mgam_feature.shape)

        sims = cosine_similarity(dual_visual, dual_text)
        # sims = cosine_sim(visual_feature, text_feature)
        return sims

def factory(opt, vocab_words, vocab_size, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    model = BaseModel(opt, vocab_words, vocab_size)
    # print(model)
    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model




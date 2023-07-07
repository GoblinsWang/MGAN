# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import utils
from PIL import Image
# from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertTokenizer
import random

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_split, tokenizer, max_seq_len=32, opt = None):

        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.loc = opt['dataset']['data_path']
        self.img_path = opt['dataset']['image_path']

        # Captions
        self.captions = []
        self.maxlength = 0

        # local features
        local_features = utils.load_from_npy(opt['dataset']['local_path'])[()]

        if data_split != 'test':
            with open(self.loc+'%s_caps_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            self.local_adj = []
            self.local_rep = []
            with open(self.loc + '%s_filename_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    # local append
                    filename = str(line.strip())[2:-1].split(".")[0] + ".txt"
                    self.local_adj.append(np.array(local_features['adj_matrix'][filename]))
                    self.local_rep.append(np.array(local_features['local_rep'][filename]))

                    self.images.append(line.strip())
        else:
            with open(self.loc + '%s_caps.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            self.local_adj = []
            self.local_rep = []
            with open(self.loc + '%s_filename.txt' % data_split, 'rb') as f:
                for line in f:
                    # local append
                    filename = str(line.strip())[2:-1].split(".")[0] + ".txt"
                    self.local_adj.append(np.array(local_features['adj_matrix'][filename]))
                    self.local_rep.append(np.array(local_features['local_rep'][filename]))

                    self.images.append(line.strip())

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "train":
            self.transform = transforms.Compose([
                # transforms.Resize((278, 278)),
                transforms.Resize((256, 256)),
                transforms.RandomRotation(degrees=(0, 90)),
                # transforms.RandomCrop(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        caption = self.captions[index]
        # target = self.get_text_input(caption)
        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)
        target = process_caption(self.tokenizer, caption_tokens)

        image = Image.open(self.img_path  +str(self.images[img_id])[2:-1]).convert('RGB')
        image = self.transform(image)  # torch.Size([3, 256, 256])

        return image, target, index, img_id

    def __len__(self):
        return self.length
    
    def get_text_input(self, caption):
        # print(caption)
        caption_tokens = self.tokenizer.tokenize(caption)
        caption_tokens = ['[CLS]'] + caption_tokens + ['[SEP]']
        caption_ids = self.tokenizer.convert_tokens_to_ids(caption_tokens)
        if len(caption_ids) >= self.max_seq_len:
            caption_ids = caption_ids[:self.max_seq_len]
        else:
            caption_ids = caption_ids + [0] * (self.max_seq_len - len(caption_ids))
        caption = torch.tensor(caption_ids).long()
        return caption

def process_caption(tokenizer, tokens, train=True):
    output_tokens = []
    deleted_idx = []
    # print(tokens)
    for i, token in enumerate(tokens):
        # print(i, token)
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
def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, captions, ids, img_ids = zip(*data)
    if len(images[0].shape) == 2:  # region feature
        # Sort a data list by caption length
        # Merge images (convert tuple of 3D tensor to 4D tensor)
        # images = torch.stack(images, 0)
        img_lengths = [len(image) for image in images]
        all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
        for i, image in enumerate(images):
            end = img_lengths[i]
            all_images[i, :end] = image[:end]
        img_lengths = torch.Tensor(img_lengths)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()

        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return all_images, img_lengths, targets, lengths, ids
    else:  # raw input image
        # Merge images (convert tuple of 3D tensor to 4D tensor)
        images = torch.stack(images, 0)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
        return images, targets, lengths, ids
    
# def collate_fn(data):
#     images, captions, ids, img_ids = zip(*data)
#     images = torch.stack(images, 0)
#     captions = torch.stack(captions, 0)

#     return images, captions, ids

def get_tokenizer(bert_path):
    # tokenizer = BertTokenizer(bert_path + 'vocab.txt')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer

def get_precomp_loader(data_split, bert_path, batch_size=100,
                       shuffle=True, num_workers=0, opt={}):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_split = data_split, tokenizer=get_tokenizer(bert_path), max_seq_len=32, opt = opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader

def get_loaders(bert_path, opt):
    train_loader = get_precomp_loader(data_split = 'train', 
                                      bert_path = bert_path,
                                      batch_size = opt['dataset']['batch_size'], 
                                      shuffle = True, 
                                      num_workers = opt['dataset']['workers'], opt=opt)
    val_loader = get_precomp_loader(data_split = 'val', 
                                    bert_path = bert_path,
                                    batch_size = opt['dataset']['batch_size'], 
                                    shuffle = False, 
                                    num_workers = opt['dataset']['workers'], opt=opt)
    return train_loader, val_loader


def get_test_loader(bert_path, opt):
    test_loader = get_precomp_loader(data_split = 'test', 
                                    bert_path = bert_path,
                                    batch_size = opt['dataset']['batch_size_val'], 
                                    shuffle = False, 
                                    num_workers = opt['dataset']['workers'], opt=opt)
    return test_loader

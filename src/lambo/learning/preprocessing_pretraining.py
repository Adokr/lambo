"""
Functions for preprocessing the data for the preprocessing LAMBO model.
"""
import random

import torch
from torch.utils.data import TensorDataset, DataLoader

from lambo.learning.preprocessing_dict import character_to_utf_feature


def encode_pretraining(documents, dictionary, maximum_length, random_seed=1):
    """
    Encode documents as neural network inputs fro pretraining
    
    :param documents: list of documents
    :param dictionary: character dictionary
    :param maximum_length: maximum length of network input
    :return: a quadruple of network input/output tensors: character encodings, UTF representations, mask vectors, true catagories
    """
    random.seed(random_seed)
    replacement_chars = [dictionary[key] for key in dictionary if len(key) == 1]
    Xchars = []
    Xutfs = []
    Xmasks = []
    Yvecs = []
    for document in documents:
        for turn in document.turns:
            Xchar = [dictionary[char] if char in dictionary else dictionary['<UNK>'] for char in turn.text]
            Xchar += [dictionary['<PAD>']] * (maximum_length - (len(Xchar) % maximum_length))
            Xutf = [character_to_utf_feature(char) for char in turn.text]
            Xutf += [character_to_utf_feature('\u0000')] * (maximum_length - (len(Xutf) % maximum_length))
            Xmask = [0] * len(Xchar)
            Yvec = Xchar.copy()
            for i in range(len(turn.text)):
                if random.random() < 0.15:
                    secondrandom = random.random()
                    if secondrandom < 0.8:
                        Xchar[i] = dictionary['<MASK>']
                    elif secondrandom < 0.9:
                        Xchar[i] = random.choice(replacement_chars)
                    else:
                        pass
                    Xmask[i] = 1
            for i in range(int(len(Xchar) / maximum_length)):
                Xchars += [Xchar[(i * maximum_length):(i * maximum_length + maximum_length)]]
                Xutfs += [Xutf[(i * maximum_length):(i * maximum_length + maximum_length)]]
                Xmasks += [Xmask[(i * maximum_length):(i * maximum_length + maximum_length)]]
                Yvecs += [Yvec[(i * maximum_length):(i * maximum_length + maximum_length)]]
    return Xchars, Xutfs, Xmasks, Yvecs


def prepare_dataloaders_pretraining(train_docs, test_docs, max_len, batch_size, dict=None):
    """
    Prapare Pytorch dataloaders for the documents (pretraining)

    :param train_docs: list of training documents
    :param test_docs: list of test documents
    :param max_len: maximum length of network input
    :param batch_size: batch size
    :param dict: character dictionary (or None, if to be created)
    :return: a triple with character dictionary, train dataloader and test dataloader
    """
    train_X_char, train_X_utf, train_X_mask, train_Y = encode_pretraining(train_docs, dict, max_len)
    train_X_char = torch.Tensor(train_X_char).to(torch.int64)
    train_X_utf = torch.Tensor(train_X_utf).to(torch.int64)
    train_X_mask = torch.Tensor(train_X_mask).to(torch.int64)
    train_Y = torch.Tensor(train_Y).to(torch.int64)
    train_dataset = TensorDataset(train_X_char, train_X_utf, train_X_mask, train_Y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_X_char, test_X_utf, test_X_mask, test_Y = encode_pretraining(test_docs, dict, max_len)
    test_X_char = torch.Tensor(test_X_char).to(torch.int64)
    test_X_utf = torch.Tensor(test_X_utf).to(torch.int64)
    test_X_mask = torch.Tensor(test_X_mask).to(torch.int64)
    test_Y = torch.Tensor(test_Y).to(torch.int64)
    test_dataset = TensorDataset(test_X_char, test_X_utf, test_X_mask, test_Y)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return dict, train_dataloader, test_dataloader

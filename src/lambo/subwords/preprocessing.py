"""
Funtions preprocessing data for subword splitting
"""
import random

import torch
from torch.utils.data import TensorDataset, DataLoader


def encode_test(text, dictionary, maximum_length):
    """
    Encode text for test purposes (no Y available).
    
    :param text: string of text to be encoded
    :param dictionary: the dictionary
    :param maximum_length: maximum length of a token (the rest will be trimmed)
    :return: a 1-tuple with a pytorch tensor of the given maximimum length
    """
    Xchar = [dictionary[char] if char in dictionary else dictionary['<UNK>'] for char in text]
    Xchar += [dictionary['<PAD>']] * (maximum_length - (len(Xchar) % maximum_length))
    Xchar = Xchar[:maximum_length]
    return torch.Tensor([Xchar]).to(torch.int64),


def encode_subwords(documents, dictionary, maximum_length):
    """
    Encode subwords as neural network inputs and outputs

    :param documents: list of documents
    :param dictionary: character dictionary
    :param maximum_length: maximum length of network input and output (the rest will be trimmed)
    :return: a pair of network input/output tensors: character encodings, true catagories (split words)
    """
    tokenCount = 0
    multiwordCount = 0
    # Count true multi-word tokens
    for document in documents:
        for turn in document.turns:
            for sentence in turn.sentences:
                for token in sentence.tokens:
                    tokenCount += 1
                    if token.is_multi_word:
                        multiwordCount += 1
    thrs = multiwordCount / tokenCount
    Xchars = []
    Ychars = []
    random.seed(1)
    for document in documents:
        for turn in document.turns:
            for sentence in turn.sentences:
                for token in sentence.tokens:
                    r = random.random()
                    if token.is_multi_word or r < thrs:
                        # Token is added to training if truly multi-word or randomly selected according to threshold
                        Xchar = [dictionary[char] if char in dictionary else dictionary['<UNK>'] for char in token.text]
                        Xchar += [dictionary['<PAD>']] * (maximum_length - (len(Xchar) % maximum_length))
                        Xchar = Xchar[:maximum_length]
                        subwords = token.subwords
                        if len(subwords) == 0:
                            subwords = [token.text]
                        Ychar = []
                        for subword in subwords:
                            if len(Ychar) != 0:
                                Ychar += [dictionary['<PAD>']]
                            Ychar += [dictionary[char] if char in dictionary else dictionary['<UNK>'] for char in
                                      subword]
                        Ychar += [dictionary['<PAD>']] * (maximum_length - (len(Ychar) % maximum_length))
                        Ychar = Ychar[:maximum_length]
                        Xchars += [Xchar]
                        Ychars += [Ychar]
    
    return Xchars, Ychars


def prepare_subwords_dataloaders(train_docs, test_docs, max_len, batch_size, dict):
    """
    Prapare Pytorch dataloaders for the documents.

    :param train_docs: list of training documents
    :param test_docs: list of test documents
    :param max_len: maximum length of network input
    :param batch_size: batch size
    :param dict: character dictionary (or None, if to be created)
    :return: a triple with character dictionary, train dataloader and test dataloader
    """
    train_X_char, train_Y = encode_subwords(train_docs, dict, max_len)
    if len(train_X_char) < 256:
        # Not enough data for training
        return None, None
    train_X_char = torch.Tensor(train_X_char).to(torch.int64)
    train_Y = torch.Tensor(train_Y).to(torch.int64)
    train_dataset = TensorDataset(train_X_char, train_Y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_X_char, test_Y = encode_subwords(test_docs, dict, max_len)
    if len(test_X_char) < 64:
        # Not enough data for testing
        return None, None
    test_X_char = torch.Tensor(test_X_char).to(torch.int64)
    test_Y = torch.Tensor(test_Y).to(torch.int64)
    test_dataset = TensorDataset(test_X_char, test_Y)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader

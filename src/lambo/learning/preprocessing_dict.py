"""
Functions for preprocessing the data for the main LAMBO model.
"""
import unicodedata

import torch, random
from torch.utils.data import TensorDataset, DataLoader

from lambo.learning.dictionary import create_dictionary

utf_category_dictionary = {
    'Co': 0,
    'Lu': 1,
    'Ll': 2,
    'Lt': 3,
    'Lm': 4,
    'Lo': 5,
    'Mn': 6,
    'Mc': 7,
    'Me': 8,
    'Nd': 9,
    'Nl': 10,
    'No': 11,
    'Pc': 12,
    'Pd': 13,
    'Ps': 14,
    'Pe': 15,
    'Pi': 16,
    'Pf': 17,
    'Po': 18,
    'Sm': 19,
    'Sc': 20,
    'Sk': 21,
    'So': 22,
    'Zs': 23,
    'Zl': 24,
    'Zp': 25,
    'Cc': 26,
    'Cf': 27,
    'Cs': 28,
    'Cn': 29
}

UNKNOWN_RATIO = 0.01
random.seed(1)


def character_to_utf_feature(char):
    """
    Convert character to one-hot UTF category vector.
    
    :param char: UTF character
    :return: one-hot feature vector
    """
    result = [0] * len(utf_category_dictionary)
    result[utf_category_dictionary[unicodedata.category(char)]] = 1
    return result


def encode_training_dict(documents, dictionary, maximum_length, sow_unknowns, finalise_all_tokens=True):
    """
    Encode documents as neural network inputs
    
    :param documents: list of documents
    :param dictionary: character dictionary
    :param maximum_length: maximum length of network input
    :param sow_unknowns: whether to randomly mask some characters as unknown
    :param finalise_all_tokens: whether every token should be properly encoded (default True, do not change)
    :return: a triple of network input/output tensors: character encodings, UTF representations, true catagories
    """
    Xchars = []
    Xutfs = []
    Yvecs = []
    for document in documents:
        offset = 0
        for turn in document.turns:
            Xchar = [dictionary[char] if char in dictionary else dictionary['<UNK>'] for char in turn.text]
            if sow_unknowns:
                pre_unknowns = sum([x == dictionary['<UNK>'] for x in Xchar])
                Xchar = [dictionary['<UNK>'] if random.random() < UNKNOWN_RATIO else xchar for xchar in Xchar]
                post_unknowns = sum([x == dictionary['<UNK>'] for x in Xchar])
                print("Sown unknowns: from every " + str(len(Xchar) / pre_unknowns) + " character to every " + str(
                    len(Xchar) / post_unknowns) + " character.")
            Xchar += [dictionary['<PAD>']] * (maximum_length - (len(Xchar) % maximum_length))
            Xutf = [character_to_utf_feature(char) for char in turn.text]
            Xutf += [character_to_utf_feature('\u0000')] * (maximum_length - (len(Xutf) % maximum_length))
            Yvec = [[0, 0, 0, 0] for x in Xchar]
            for sentence in turn.sentences:
                for tokenI in range(len(sentence.tokens)):
                    token = sentence.tokens[tokenI]
                    for i in range(token.end - token.begin):
                        Yvec[token.begin + i - offset][0] = 1
                    if finalise_all_tokens or (
                            tokenI + 1 != len(sentence.tokens) and token.end == sentence.tokens[tokenI + 1].begin):
                        Yvec[token.end - 1 - offset][1] = 1
                    if token.is_multi_word:
                        Yvec[token.end - 1 - offset][2] = 1
                Yvec[sentence.tokens[-1].end - 1 - offset][3] = 1
            offset = offset + len(turn.text)
            for i in range(int(len(Xchar) / maximum_length)):
                Xchars += [Xchar[(i * maximum_length):(i * maximum_length + maximum_length)]]
                Xutfs += [Xutf[(i * maximum_length):(i * maximum_length + maximum_length)]]
                Yvecs += [Yvec[(i * maximum_length):(i * maximum_length + maximum_length)]]
    return Xchars, Xutfs, Yvecs


def prepare_test_withdict(text, dictionary, maximum_length):
    """
    Encode test documents as neural network inputs. The same as ``encode_training_dict``, but with no true category output.
    
    :param text: text to be encoded
    :param dictionary: character dictionary
    :param maximum_length: maximum length of network input
    :return: a pair of network input tensors: character encodings, UTF representations
    """
    Xchars = []
    Xutfs = []
    Xchar = [dictionary[char] if char in dictionary else dictionary['<UNK>'] for char in text]
    Xchar += [dictionary['<PAD>']] * (maximum_length - (len(Xchar) % maximum_length))
    Xutf = [character_to_utf_feature(char) for char in text]
    Xutf += [character_to_utf_feature('\u0000')] * (maximum_length - (len(Xutf) % maximum_length))
    for i in range(int(len(Xchar) / maximum_length)):
        Xchars += [Xchar[(i * maximum_length):(i * maximum_length + maximum_length)]]
        Xutfs += [Xutf[(i * maximum_length):(i * maximum_length + maximum_length)]]
    return torch.Tensor(Xchars).to(torch.int64), torch.Tensor(Xutfs).to(torch.int64)


def prepare_dataloaders_withdict(train_docs, test_docs, max_len, batch_size, sow_unknowns, dict=None):
    """
    Prapare Pytorch dataloaders for the documents.
    
    :param train_docs: list of training documents
    :param test_docs: list of test documents
    :param max_len: maximum length of network input
    :param batch_size: batch size
    :param sow_unknowns: whether to randomly mask some characters as unknown
    :param dict: character dictionary (or None, if to be created)
    :return: a triple with character dictionary, train dataloader and test dataloader
    """
    if dict is None:
        dict = create_dictionary([doc.text for doc in train_docs])
    train_X_char, train_X_utf, train_Y = encode_training_dict(train_docs, dict, max_len, sow_unknowns)
    train_X_char = torch.Tensor(train_X_char).to(torch.int64)
    train_X_utf = torch.Tensor(train_X_utf).to(torch.int64)
    train_Y = torch.Tensor(train_Y).to(torch.int64)
    train_dataset = TensorDataset(train_X_char, train_X_utf, train_Y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_X_char, test_X_utf, test_Y = encode_training_dict(test_docs, dict, max_len, False)
    test_X_char = torch.Tensor(test_X_char).to(torch.int64)
    test_X_utf = torch.Tensor(test_X_utf).to(torch.int64)
    test_Y = torch.Tensor(test_Y).to(torch.int64)
    test_dataset = TensorDataset(test_X_char, test_X_utf, test_Y)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return dict, train_dataloader, test_dataloader

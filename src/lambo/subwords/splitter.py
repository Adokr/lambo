import torch

from lambo.segmenter.lambo import Lambo
from lambo.subwords.preprocessing import encode_test


class LamboSplitter():
    """
    Class for splitting tokens into sub-words (wrapper for neural network)
    """
    
    @classmethod
    def from_path(cls, model_path, model_name):
        """
        Obtain a LAMBO subword splitter by reading a model from a given path.

        :param model_path: directory including the model files
        :param model_name: model name
        :return: instance of LamboSplitter
        """
        model = torch.load(model_path / (model_name + '_subwords.pth'), map_location=torch.device('cpu'))
        dict = Lambo.read_dict(model_path / (model_name + '.dict'))
        return cls(model, dict)
    
    def __init__(self, model, dict):
        """
        Create a new LAMBO subword splitter from a given model and dictionary.

        :param model: prediction Pytorch model
        :param dict: dictionary
        """
        self.model = model
        self.dict = dict
        self.inv_dict = {dict[key]: key for key in dict}
    
    def split(self, token_text):
        """
        Split a given token text
        
        :param token_text: string with a token to split
        :return: list of subwords
        """
        # Too long for the maximum length
        if len(token_text) >= self.model.max_len:
            return [token_text]
        Xs = encode_test(token_text, self.dict, self.model.max_len)
        with torch.no_grad():
            Y = self.model(*Xs)
        codes = Y.argmax(2).numpy()[0]
        decisions = [self.inv_dict[code] for code in codes]
        # Recover the subwords from the network output
        result = ['']
        for char in decisions:
            if len(char) == 1:
                result[-1] += char
            elif char == '<PAD>':
                if result[-1] == '':
                    break
                result.append('')
            else:
                return [token_text]
        result = [subword for subword in result if subword != '']
        if len(result) == 0:
            return [token_text]
        return result

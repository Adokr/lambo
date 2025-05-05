import re

import importlib.resources as resources
import torch

from lambo.data.document import Document
from lambo.data.sentence import Sentence
from lambo.data.token import Token
from lambo.data.turn import Turn
from lambo.learning.preprocessing_dict import prepare_test_withdict
from lambo.utils.download import download_model, default_type
from lambo.utils.special_tokens import detect_special_tokens

# Reading the turn separator file
turn_sep_file_str = resources.read_text('lambo.resources', 'turn_regexp.txt', encoding='utf-8', errors='strict')
turnseps = [line for line in turn_sep_file_str.split('\n') if len(line) > 0]
turnseps = sorted(turnseps, key=len, reverse=True)


class Lambo():
    """
    LAMBO segmenter.
    """
    
    @classmethod
    def get(cls, provided_name, with_splitter=True, device=torch.device('cpu')):
        """
        Obtain a LAMBO segmenter based on the name of the model.
        
        :param provided_name: either a full model name (``LAMBO_no_pretraining-UD_Polish-PDB``), or language name (``Polish``) or ISO 639-1 code (``pl``)
        :param with_splitter: should a subword splitter be loaded as well
        :param device: pytorch device to use for inference
        :return: LAMBO segmenter based on the expected model
        """
        if '-' in provided_name:
            # It's s regular name
            model_name = provided_name
        else:
            # It's an alias -- language name or code
            model_name = Lambo.getDefaultModel(provided_name)
        dict_path, model_path, splitter_path = download_model(model_name)
        dict = Lambo.read_dict(dict_path)
        model = torch.load(model_path, map_location=torch.device('cpu'))
        splitter = None
        if with_splitter and splitter_path:
            from lambo.subwords.splitter import LamboSplitter
            splitter = LamboSplitter.from_path(splitter_path.parent, model_name)
        return cls(model, dict, splitter, device)
    
    @staticmethod
    def getDefaultModel(provided_name):
        """
        Get the default model for the given language. If more than one variant is available for a given language, the one with an asterisk in ``resources/languages.txt`` is considered default. The default model type is ``LAMBO``.
        
        :param provided_name: language, specified either as name (``Polish``) or ISO 639-1 code (``pl``)
        :return: full model name
        """
        treebank = None
        for line in resources.read_text('lambo.resources', 'languages.txt', encoding='utf-8', errors='strict').split(
                '\n'):
            if line[0] == '#':
                continue
            parts = line.split(' ')
            if provided_name == parts[1] or provided_name == parts[2]:
                if treebank is None or (len(parts) >= 4 and parts[3] == '*'):
                    # if more than one treebank for a language available, choose the one with asterisk
                    treebank = parts[0]
        if treebank is None:
            raise ValueError("Unrecognised language name or code: '" + provided_name + "'")
        model_name = default_type + '-' + treebank
        print("Using model " + model_name)
        return model_name
    
    @classmethod
    def from_path(cls, model_path, model_name, with_splitter=True, device=torch.device('cpu')):
        """
        Obtain a LAMBO segmenter by reading a model from a given path.
        
        :param model_path: directory including the model files
        :param model_name: model name
        :param device: pytorch device to use for inference
        :return:
        """
        model = torch.load(model_path / (model_name + '.pth'), map_location=torch.device('cpu'))
        dict = Lambo.read_dict(model_path / (model_name + '.dict'))
        splitter = None
        if with_splitter and (model_path / (model_name + '_subwords.pth')).exists():
            from lambo.subwords.splitter import LamboSplitter
            splitter = LamboSplitter.from_path(model_path, model_name)
        return cls(model, dict, splitter, device)
    
    def __init__(self, model, dict, splitter=None, device=torch.device('cpu')):
        """
        Create a new LAMBO segmenter from a given model and dictionary.
        
        :param model: prediction Pytorch model
        :param dict: dictionary
        :param device: pytorch device to use for inference
        """
        self.model = model
        self.dict = dict
        self.splitter = splitter
        self.device = device
        self.model.to(self.device)
    
    @staticmethod
    def read_dict(dict_path):
        """
        Read character dictionary from a given path.
        
        :param dict_path: path to the .dict file
        :return: character dictionary
        """
        dict = {}
        prevEmpty = False
        # to properly handle non-standard newline characters
        chunks = dict_path.read_bytes().decode('utf-8').split('\n')
        for chunk in chunks:
            if chunk == '':
                prevEmpty = True
                continue
            parts = chunk.split('\t')
            if len(parts) == 3 and parts[0] == '' and parts[1] == '':
                # TAB character
                parts = ['\t', parts[2]]
            # to properly handle newline characters in the dictionary
            if parts[0] == '' and prevEmpty:
                parts[0] = '\n'
            if parts[0] in dict:
                print("WARNING: duplicated key in dictionary")
            dict[parts[0]] = int(parts[1])
            prevEmpty = False
        return dict
    
    @staticmethod
    def slice_on_special_tokens(spans, decision):
        """
        Modify the decision tensor so that the special tokens (emojis and pauses) are preserved as separate and undivided tokens
        
        :param spans: list of (begin,end) pairs of special tokens
        :param decision: the original decision tensor
        :return: the modified decision tensor
        """
        for begin, end in spans:
            # All characters within span are in-token
            for i in range(begin, end):
                decision[i][0] = 1
            # The last character ends the token
            for i in range(begin, end - 1):
                decision[i][1] = 0
            decision[end - 1][1] = 1
            # No multi-word tokens here
            for i in range(begin, end):
                decision[i][2] = 0
            # If a sentence ends here, move to the span end
            if any([decision[i][3] for i in range(begin, end)]):
                for i in range(begin, end - 1):
                    decision[i][3] = 0
                decision[end - 1][3] = 1
            # End the preceding token
            if begin != 0 and decision[begin - 1][0]:
                decision[begin - 1][1] = 1
    
    def segment(self, text):
        """
        Perform the segmentation of the text. This involves:
        
        * splitting the document into turns using turn markers from ``turn_regexp.txt``
        * splitting the turns into sentences and tokens according to the model's predictions (including splitting into subwords)
        * modifying the output to account for special tokens (emojis and pauses)
        
        :param text: input text
        :return: segmented document
        """
        document = Document()
        document.set_text(text)
        regexp = '(' + '|'.join(['(?:' + x + ')' for x in turnseps]) + ')'
        parts = re.split(regexp, text) + ['']
        turn_texts = [parts[i * 2] + parts[i * 2 + 1] for i in range(int(len(parts) / 2))]
        turn_offset = 0
        for turn_text in turn_texts:
            turn = self.segment_turn(turn_offset, turn_text)
            document.add_turn(turn)
            turn_offset += len(turn.text)
        assert document.check_sanity()
        return document
    
    def segment_turn(self, turn_offset, text):
        """
        Perform the segmentation of a single turn.
        
        :param turn_offset: where does this turn start (used to compute token coordinates)
        :param text: turn text
        :return: a segmented turn
        """
        
        # Prepare neural network input
        X = prepare_test_withdict(text, self.dict, self.model.max_len)
        
        # compute neural network output
        with torch.no_grad():
            X = [x.to(self.device) for x in X]
            Y = self.model(*X)
        Y = Y.to('cpu')
        
        # perform postprocessing
        decisions = self.model.postprocessing(Y, text)
        
        # detect special tokens and slice on them
        special_tokens = detect_special_tokens(text)
        self.slice_on_special_tokens(special_tokens, decisions)
        
        turn = Turn()
        turn.set_text(text)
        sentence = Sentence()
        token_begin = -1
        sentence_begin = 0
        i_forward = 0
        for i in range(len(text)):
            # Obtain the decision for this position
            in_token, token_end, mwtoken_end, sentence_end = decisions[i]
            # Skip if these characters added to previous sentence
            if i < i_forward:
                continue
            # Sentence ends if end of text
            if i == len(text) - 1:
                sentence_end = 1
            # Token ends if next character is not in a token
            elif in_token and (not decisions[i + 1][0]):
                token_end = 1
            # Token ends if mwtoken or sentence does
            if sentence_end or mwtoken_end:
                token_end = 1
            # Token can only end if in token
            if token_end:
                in_token = 1
            # Interpretation
            if token_begin == -1 and in_token:
                # New token
                token_begin = i
            if token_end:
                # End of token
                token = Token(turn_offset + token_begin, turn_offset + i + 1, text[token_begin:(i + 1)], mwtoken_end)
                if mwtoken_end and self.splitter:
                    # If token looks multi-word and splitter is avilable, use it
                    subwords = self.splitter.split(token.text)
                    if len(subwords) == 1:
                        # If not split in the end, ignore
                        token.is_multi_word = False
                    else:
                        token.subwords = subwords
                sentence.add_token(token)
                token_begin = -1
            if sentence_end:
                # End of sentence
                # Look for beginning of next
                i_forward = i + 1
                while i_forward < len(text) and decisions[i_forward][0] == 0 and decisions[i_forward][1] == 0 and \
                        decisions[i_forward][2] == 0 and decisions[i_forward][3] == 0:
                    # First non-white character
                    i_forward = i_forward + 1
                sentence_text = text[sentence_begin:i_forward]
                sentence.set_text(sentence_text)
                turn.add_sentence(sentence)
                sentence = Sentence()
                sentence_begin = i_forward
        return turn
    
    @staticmethod
    def name():
        """ Return the name of the segmenter
        
        :return: ``LAMBO``
        """
        return 'LAMBO'

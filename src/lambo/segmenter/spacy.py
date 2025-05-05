from textwrap import wrap

import spacy

from lambo.data.document import Document
from lambo.data.sentence import Sentence
from lambo.data.token import Token
from lambo.data.turn import Turn


class Spacy_segmenter():
    """
    Segmenter based on spaCy, implemented as a baseline for performance comparison with LAMBO.
    """
    
    def __init__(self):
        """
        Initialise the segmenter. As in COMBO, ``en_cor_web_sm`` model is used.
        """
        self.nlp = spacy.load("en_core_web_sm")
    
    def segment(self, text):
        """
        Segment a text using spaCy. Sentences and token are divided according to the model output. No multi-word tokens are included. All the sentences belong to the same turn.
        
        :param text: input text string
        :return: output document
        """
        result = Document()
        result.set_text(text)
        turn = Turn()
        turn.set_text(text)
        result.add_turn(turn)
        offset = 0
        # Prevent 'too long text' error by splitting the input by 100 000 characters.
        MAX_LEN = 100000
        for sub_text in wrap(text, MAX_LEN, drop_whitespace=False):
            doc = self.nlp(sub_text)
            for spacy_sentence in doc.sents:
                sentence = Sentence()
                for spacy_token in doc[spacy_sentence.start:spacy_sentence.end]:
                    token = Token(offset + spacy_token.idx, offset + spacy_token.idx + len(spacy_token.text),
                                  spacy_token.text, False)
                    sentence.add_token(token)
                sentence_begin = doc[spacy_sentence.start].idx
                if spacy_sentence.end == len(doc):
                    sentence_end = len(sub_text)
                else:
                    sentence_end = doc[spacy_sentence.end].idx
                sentence.text = sub_text[sentence_begin:sentence_end]
                turn.add_sentence(sentence)
            offset += len(sub_text)
        assert result.check_sanity()
        return result
    
    def name(self):
        """
        Name of the tokenizer.
        
        :return: ``spaCy``
        """
        return 'spaCy'

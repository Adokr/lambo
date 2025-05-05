"""
A collection of functions used to read annotated Universal Dependencies files (.conllu format). See the `UD documentation <https://universaldependencies.org/conll18/evaluation.html>`_ for details.
"""
import random

from lambo.data.document import Document
from lambo.data.sentence import Sentence
from lambo.data.token import Token
from lambo.data.turn import Turn

# Taken from https://www.lesinskis.com/python-unicode-whitespace.html
UNICODE_WHITESPACE_CHARACTERS = [
    "\u0009",  # character tabulation
    "\u000a",  # line feed
    "\u000b",  # line tabulation
    "\u000c",  # form feed
    "\u000d",  # carriage return
    "\u0020",  # space
    "\u0085",  # next line
    "\u00a0",  # no-break space
    "\u1680",  # ogham space mark
    "\u2000",  # en quad
    "\u2001",  # em quad
    "\u2002",  # en space
    "\u2003",  # em space
    "\u2004",  # three-per-em space
    "\u2005",  # four-per-em space
    "\u2006",  # six-per-em space
    "\u2007",  # figure space
    "\u2008",  # punctuation space
    "\u2009",  # thin space
    "\u200A",  # hair space
    "\u2028",  # line separator
    "\u2029",  # paragraph separator
    "\u202f",  # narrow no-break space
    "\u205f",  # medium mathematical space
    "\u3000",  # ideographic space
]


def read_treebank(dir_path, random_separators):
    """
    Function to read an annotated treebank, e.g. from a UD release. It only takes into account segmentation, ignoring
    the dependency parsing information, and expects separate training, development and test files.
    
    :param dir_path: path to the directory with annotated files, including ``XXXX-ud-train.conllu``, ``XXXX-ud-dev.conllu`` and ``XXXX-ud-train.conllu``.
    :param random_separators: whether random separators between tokens should be added when reconstructing text
    :return: a triple of documents, corresponding to training, development and testing subsets.
    """
    train_path = None
    dev_path = None
    test_path = None
    for path in dir_path.iterdir():
        if str(path).endswith('-ud-train.conllu'):
            train_path = path
        elif str(path).endswith('-ud-dev.conllu'):
            dev_path = path
        elif str(path).endswith('-ud-test.conllu'):
            test_path = path
    return read_document(train_path, random_separators), read_document(dev_path, random_separators), read_document(
        test_path, random_separators)


def word_separator(randomly):
    """
    Generates separator between tokens.
    
    :param randomly: should it be done randomly instead of a single space
    :return: separator text
    """
    if not randomly:
        # just a single space
        return ' '
    indicator = random.random()
    if indicator < 0.95:
        # usually a single space
        return ' '
    elif indicator < 0.96:
        # 1% probability of newline
        return '\n'
    elif indicator < 0.98:
        # 2% proability of double spearator
        return word_separator(True) + word_separator(True)
    else:
        # 2% probability of random whitespace character
        return random.choice(UNICODE_WHITESPACE_CHARACTERS)


def sentence_separator(randomly):
    """
    Generates separator between sentences.
    
    :param randomly: should it be done randomly instead of a single newline
    :return: separator text
    """
    if not randomly:
        # just a single newline
        return '\n'
    indicator = random.random()
    if indicator < 0.8:
        # 80% probability of a single space
        return ' '
    elif indicator < 0.9:
        # 10% probability of newline
        return '\n'
    elif indicator < 0.98:
        # 8% probability of double separator
        return sentence_separator(True) + sentence_separator(True)
    else:
        # 2% probability of random whitespace character
        return random.choice(UNICODE_WHITESPACE_CHARACTERS)


def read_document(file_path, random_separators):
    """
    Reads a single document from .conllu
    
    :param file_path: path to .conllu file
    :param random_separators: whether random separators should be used
    :return: an instance of Document
    """
    if file_path is None:
        return None
    turn = Turn()
    turn_text = ""
    sentence = Sentence()
    sentence_text = ""
    word_range = [0, 0]
    current_offset = 0
    separator = ''
    lastToken = None
    for line in file_path.read_text().split('\n'):
        if line.startswith('#'):
            # Comment, ignore
            pass
        elif line == '':
            # End of sentence
            if sentence_text == '':
                continue
            # Replacing word separator with sentence separator
            if len(separator) > 0:
                sentence_text = sentence_text[0:(-1 * len(separator))]
                current_offset -= len(separator)
                separator = sentence_separator(random_separators)
                sentence_text += separator
                current_offset += len(separator)
            # Saving the sentence
            sentence.set_text(sentence_text)
            turn_text += sentence_text
            turn.add_sentence(sentence)
            sentence = Sentence()
            sentence_text = ""
            word_range = [0, 0]
        else:
            parts = line.split('\t')
            is_copy = any(x.startswith('CopyOf=') for x in parts[-1].split('|')) or ('.' in parts[0])
            if is_copy:
                continue
            numbers = [int(x) for x in parts[0].split('-')]
            form = parts[1]
            space_after_no = ('SpaceAfter=No' in parts[-1].split('|'))
            if len(numbers) == 1:
                if word_range[0] <= numbers[0] <= word_range[1]:
                    # Individual word within multi-word token
                    lastToken.addSubword(form)
                else:
                    # Individual word not covered
                    token = Token(current_offset, current_offset + len(form), form, False)
                    sentence_text += form
                    current_offset += len(form)
                    if space_after_no:
                        separator = ''
                    else:
                        separator = word_separator(random_separators)
                    sentence_text += separator
                    current_offset += len(separator)
                    sentence.add_token(token)
            elif len(numbers) == 2:
                # Multi-word token
                token = Token(current_offset, current_offset + len(form), form, True)
                sentence_text += form
                current_offset += len(form)
                if space_after_no:
                    separator = ''
                else:
                    separator = word_separator(random_separators)
                sentence_text += separator
                current_offset += len(separator)
                sentence.add_token(token)
                word_range = numbers
                lastToken = token
    turn.set_text(turn_text)
    document = Document()
    document.set_text(turn_text)
    document.add_turn(turn)
    
    assert document.check_sanity()
    return document

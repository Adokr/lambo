"""
Detects special tokens, i.e. text fragments that should always be included in a spearate token. Two types of tokens are handled:
* Emojis, using dictionary from Morfeusz SGJP (29.05.2022), http://morfeusz.sgjp.pl/download/, saved in ``resources/emoji.tab``
* Pauses, denoting non-verbal sounds noted in the speech transcript, using dictionary in ``resources/pauses.txt``
"""
import re

import importlib.resources as resources

emoji_file_str = resources.read_text('lambo.resources', 'emoji.tab', encoding='utf-8', errors='strict')
emojis = [line.split('\t')[0] for line in emoji_file_str.split('\n') if len(line) > 0]
pause_file_str = resources.read_text('lambo.resources', 'pauses.txt', encoding='utf-8', errors='strict')
pauses = [line for line in pause_file_str.split('\n') if len(line) > 0]
special_tokens = sorted(emojis + pauses, key=len, reverse=True)
token_pattern = re.compile(r'(?:{})'.format('|'.join(map(re.escape, special_tokens))))


def detect_special_tokens(text):
    """
    Detects special tokens.
    
    :param text: input text string
    :return: list of tuples, each containing beginning and end of a found special token
    """
    result = [(match.start(), match.end()) for match in token_pattern.finditer(text)]
    return result

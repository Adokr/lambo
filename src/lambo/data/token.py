class Token:
    """
    Class representing a token, i.e. a short piece of text, included in a sentence. Usually, tokens correspond to words,
    but multi-word tokens are possible.
    """
    
    def __init__(self, begin, end, text, is_multi_word):
        """
        Create a new token.
        
        :param begin: the offset of the beginning of the text covered by the token with respect to the document's text
        :param end: the offset of the end of the text covered by the token (first character after the token) with
        respect to the document's text
        :param text: text covered by the token
        :param is_multi_word: is this a multi-word token
        """
        self.begin = begin
        self.end = end
        self.text = text
        self.is_multi_word = is_multi_word
        self.subwords = []
        
    def addSubword(self, subword):
        """
        Add a subword to the token.
        
        :param subword: the text of the subword to add
        """
        self.subwords.append(subword)

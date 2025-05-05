class Sentence:
    """
    Class representing a sentence. Not that the division of turn into sentences needs to be complete, i.e. a
    concatenation of the text of all sentences in a turn should equal this turn's text.
    """
    def __init__(self):
        """
        Create a new sentence, with no text or tokens.
        """
        self.tokens = []
        self.text = None
    
    def set_text(self, text):
        """
        Assign a text string to the sentence.
        
        :param text: text string to be associated with the sentence
        :return: no value returned
        """
        self.text = text
    
    def add_token(self, token):
        """
        Add a token to the sentence.
        
        :param token: the token to be added
        :return: no value returned
        """
        self.tokens.append(token)

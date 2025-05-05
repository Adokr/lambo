class Turn:
    """ Class representing a turn. This is designed to correspond to speech uttered by one of the participants in
    transcriptions of dialogue, but can also represent any other high-level section of text, e.g. a paragraph. Not that
    the division of document into sentences turn to be complete, i.e. a concatenation of the text of all turns in a
    document should equal this document's text."""
    
    def __init__(self):
        """
        Create a new turn, with no text or sentences.
        """
        self.sentences = []
        self.text = None
    
    def set_text(self, text):
        """
        Assign a text string to the turn.
        
        :param text: text string to be associated with the turn
        :return: no value returned
        """
        self.text = text
    
    def add_sentence(self, sentence):
        """
        Add a sentence to the turn.
        
        :param sentence: the sentence to be added
        :return: no value returned
        """
        self.sentences.append(sentence)

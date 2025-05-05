from lambo.utils.printer import print_document_to_screen


class Document:
    """
    Class representing a single textual document. Each document is a text string, divided into turns, divided into sentences,
    including tokens.
    """
    
    def __init__(self):
        """
        Create a new document, with no text and no turns.
        """
        self.text = None
        self.turns = []
    
    def set_text(self, text):
        """
        Assign a text string to the document.
        
        :param text: text string to be associated with this document
        :return: no value returned
        """
        self.text = text
    
    def add_turn(self, turn):
        """
        Add a turn to the document.
        
        :param turn: turn to be added
        :return: no value returned
        """
        self.turns.append(turn)
    
    def check_sanity(self):
        """
        Check if the document is properly defined. The concatenation of text covered by all turns should equal the
        document text and the concatenation of text of all sentences in a turn should equal the turn text. Finally, the
        text of each token should be an exact sub-string of the sentence text according to the provided offsets.
        
        :return: True if document is well-defined, False otherwise
        """
        if ''.join([turn.text for turn in self.turns]) != self.text:
            return False
        for turn in self.turns:
            if ''.join([sentence.text for sentence in turn.sentences]) != turn.text:
                return False
            for sentence in turn.sentences:
                for token in sentence.tokens:
                    if self.text[token.begin:token.end] != token.text:
                        return False
        return True
    
    def print(self):
        """
        Print human-readable document representation to the standard output.
        
        :return: no value returned
        """
        print_document_to_screen(self)

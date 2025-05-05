def create_dictionary(texts, with_mask=False, MIN_OCC=10):
    """
    Create a dictionary of characters in a given language. Dictionary is indexed by a character string and includes the integer codes for encoding them. Three special codes are reserved:
    
    * ``<PAD>`` -- for padding
    * ``<UNK>`` -- for unkknown characters
    * ``<MASK>`` -- for masked characters (optional)
    
    :param texts: list of text strings, based on which a dictionary will be created
    :param with_mask: whether a mask token ID should be reserved (used for pretraining)
    :param MIN_OCC: minimum number of occurrences of a character that has a code in the dictionary (default=10)
    :return: the created dictionary
    """
    occurrences = {}
    for text in texts:
        for char in text:
            if char not in occurrences:
                occurrences[char] = 1
            else:
                occurrences[char] = occurrences[char] + 1
    dictionary = {'<PAD>': 0, '<UNK>': 1}
    if with_mask:
        dictionary['<MASK>'] = 2
    for char in occurrences:
        if occurrences[char] >= MIN_OCC:
            dictionary[char] = len(dictionary)
    return dictionary

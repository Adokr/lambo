"""
Collection of functions for converting annotations to text
"""

def print_document_to_screen(document):
    """
    Prints a given document to the screen
    
    :param document: input document
    :return: no value returned
    """
    print('======= DOCUMENT =======')
    print('TEXT: ' + document.text[:100] + '...')
    for turn in document.turns:
        print('======= TURN =======')
        print('TEXT: ' + turn.text[:100] + '...')
        for sentence in turn.sentences:
            print('======= SENTENCE =======')
            print('TEXT: "' + sentence.text + '"')
            formatted = ''
            for token in sentence.tokens:
                if token.is_multi_word:
                    formatted += '(' + token.text+ '=' + '-'.join(token.subwords) + ')'
                else:
                    formatted += '(' + token.text + ')'
            print('TOKENS: ' + formatted)


def print_document_to_conll(document, path):
    """
    Prints a given document to a .conllu file
    
    :param document: input document
    :param path: path to the output .conllu file
    :return: no value returned
    """
    sentence_id = 1
    with open(path, "w") as file1:
        for turn in document.turns:
            for sentence in turn.sentences:
                if sentence.text.strip() == '':
                    continue
                file1.write('# sent_id = out-' + str(sentence_id) + '\n')
                sentence_id += 1
                file1.write('# text = ' + sentence.text + '\n')
                token_id = 1
                for token in sentence.tokens:
                    token_text = token_text_with_whitespace_for_conllu(token, document, turn, sentence).strip()
                    if token_text == '':
                        continue
                    if token.is_multi_word and len(token.subwords) > 1:
                        file1.write(str(token_id))
                        file1.write('-' + str(token_id + len(token.subwords) - 1))
                        file1.write('\t' + token_text + '\t_\t_\t_\t_\t_\t_\t_\t_\n')
                        for word in token.subwords:
                            file1.write(str(token_id) + '\t' + word + '\t_\t_\t_\t_\t' + str(token_id - 1) + '\t_\t_\t_\n')
                            token_id += 1
                    else:
                        file1.write(str(token_id))
                        file1.write('\t' + token_text + '\t_\t_\t_\t_\t' + str(token_id - 1) + '\t_\t_\t_\n')
                        token_id += 1
                file1.write('\n')


def token_text_with_whitespace_for_conllu(this_token, document, this_turn, this_sentence):
    """
    Retrieves a token text, but including the characters (likely whitespace) separating it from the next token.
    
    :param this_token: the text of which token is to be retrieved
    :param document: in which document the token resides
    :param this_turn: in which turn the token resides
    :param this_sentence: in which sentence the token resides
    :return: the text between the beginning of this token and the next one (or end of document)
    """
    next_token = None
    searching = False
    for turn in document.turns:
        if (not searching) and (turn is not this_turn):
            continue
        for sentence in turn.sentences:
            if (not searching) and (sentence is not this_sentence):
                continue
            for token in sentence.tokens:
                if token is this_token:
                    searching = True
                elif searching:
                    next_token = token
                    break
            if next_token:
                break
        if next_token:
            break
    if next_token:
        return document.text[this_token.begin:next_token.begin]
    else:
        return document.text[this_token.begin:]

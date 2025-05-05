"""
Functions used to obtain multilingual corpora from `OSCAR <https://oscar-corpus.com/>`_
"""
import json
import random
import urllib
import time
from urllib.error import HTTPError

from lambo.data.document import Document
from lambo.data.turn import Turn


def download_archive1_from_oscar(language, path, OSCAR_LOGIN, OSCAR_PASSWORD, retry=6):
    """
    Download a corpus for a given language from OSCAR. If many archives are available, only the first one is taken into account.
    
    :param language: what language should be used (ISO 639-1 code)
    :param path: where to save the corpus to
    :param OSCAR_LOGIN: login to access OSCAR
    :param OSCAR_PASSWORD: password to access OSCAR
    :param retry: how many times retry after getting an error
    :return: no value returned
    """
    url_part1 = 'https://oscar-prive.huma-num.fr/2201/packaged/' + language + '_meta/' + language + '_meta_part_1.jsonl.gz'
    url_all = 'https://oscar-prive.huma-num.fr/2201/packaged/' + language + '_meta/' + language + '_meta.jsonl.gz'
    passman = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    passman.add_password(None, url_all, OSCAR_LOGIN, OSCAR_PASSWORD)
    passman.add_password(None, url_part1, OSCAR_LOGIN, OSCAR_PASSWORD)
    authhandler = urllib.request.HTTPBasicAuthHandler(passman)
    opener = urllib.request.build_opener(authhandler)
    urllib.request.install_opener(opener)
    for i in range(retry):
        error = None
        try:
            urllib.request.urlretrieve(url_all, path)
            return
        except HTTPError as err:
            if err.code == 404:
                print("Multi-part archive, downloading part 1.")
            else:
                error = err
        # If error was not saved, this must've been 404. Assume that's because of multi-part archive.
        if error is None:
            try:
                urllib.request.urlretrieve(url_part1, path)
                return
            except HTTPError as err:
                error = err
        if i == retry - 1 or error.code<500:
            raise error
        secs = ((i + 1) * (i + 1) * (i + 1) * 15)
        print("[Got " + str(error.code) + ", retrying after " + str(secs) + " seconds...]")
        time.sleep(secs)


def read_jsonl_to_documents(fileobj, MAX_LEN=3000000):
    """
    Read the JSONL file retrieved from OSCAR to documents. The text is randomly divided into training and test subsets.
    
    :param fileobj: file object with the downloaded OSCAR corpus in JSONL format
    :param MAX_LEN: maximum length of a single document -- longer ones will be divided
    :return: pair of lists, containing train and test documents
    """
    train_documents = []
    test_documents = []
    train_document = Document()
    test_document = Document()
    length = 0
    for line in fileobj:
        struct = json.loads(line.strip())
        content = struct['content']
        pseudo_turn = Turn()
        pseudo_turn.set_text(content)
        if random.random() < 0.2:
            test_document.add_turn(pseudo_turn)
        else:
            train_document.add_turn(pseudo_turn)
            length += len(pseudo_turn.text)
        if length >= MAX_LEN:
            train_documents.append(train_document)
            test_documents.append(test_document)
            train_document = Document()
            test_document = Document()
            length = 0
    if len(train_document.turns) > 0 and len(test_document.turns) > 0:
        train_documents.append(train_document)
        test_documents.append(test_document)
    return train_documents, test_documents

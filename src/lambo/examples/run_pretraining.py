"""
Script from pretraining models using OSCAR corpora
"""
import gzip
from urllib.error import HTTPError

import importlib.resources as resources
from pathlib import Path

import torch
import sys

from lambo.learning.dictionary import create_dictionary
from lambo.learning.model_pretraining import LamboPretrainingNetwork
from lambo.learning.preprocessing_dict import utf_category_dictionary
from lambo.learning.preprocessing_pretraining import encode_pretraining, prepare_dataloaders_pretraining
from lambo.learning.train import pretrain
from lambo.utils.oscar import read_jsonl_to_documents, download_archive1_from_oscar

if __name__ == '__main__':
    outpath = Path(sys.argv[1])  # Path.home() / 'PATH-TO/models/pretrained/'
    tmppath = Path(sys.argv[2])  # Path.home() / 'PATH-TO/tmp/tmp.jsonl.gz'
    # These need to be filled ine before running. OSCAR is avaialable on request.
    OSCAR_LOGIN = sys.argv[3]
    OSCAR_PASSWORD = sys.argv[4]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    languages_file_str = resources.read_text('lambo.resources', 'languages.txt', encoding='utf-8', errors='strict')
    languages = [line.split(' ')[1] for line in languages_file_str.split('\n') if
                 line[0] != '#' and line.split(' ')[1] != '?']
    languages = list(dict.fromkeys(languages))
    
    MAX_DOCUMENTS = 100
    CONTEXT_LEN = 1024
    
    for l, language in enumerate(languages):
        if l % 5 != int(sys.argv[5]):
            continue
        if (outpath / ('oscar_' + language + '.pth')).exists():
            continue
        print("Language: " + language)
        print("Downloading corpus...")
        try:
            download_archive1_from_oscar(language, tmppath, OSCAR_LOGIN, OSCAR_PASSWORD)
        except HTTPError as err:
            if err.code==404:
                print("Language unavailable in OSCAR. moving on...")
                continue
            else:
                raise err
        with gzip.open(tmppath) as jsonfile:
            train_documents, test_documents = read_jsonl_to_documents(jsonfile)
        print("Generated " + str(len(train_documents)) + " documents.")
        dict = None
        model = None
        for i, (document_train, document_test) in enumerate(zip(train_documents, test_documents)):
            if i == MAX_DOCUMENTS:
                break
            if dict is None:
                dict = create_dictionary([turn.text for turn in document_train.turns], with_mask=True)
                model = LamboPretrainingNetwork(CONTEXT_LEN, dict, len(utf_category_dictionary))
            print(str(i + 1) + '/' + str(min(len(train_documents), MAX_DOCUMENTS)))
            Xchars, Xutfs, Xmasks, Yvecs = encode_pretraining([document_train], dict, CONTEXT_LEN)
            _, train_dataloader, test_dataloader = prepare_dataloaders_pretraining([document_train],
                                                                                   [document_test], CONTEXT_LEN, 32,
                                                                                   dict)
            pretrain(model, train_dataloader, test_dataloader, 1, device)
        torch.save(model, outpath / ('oscar_' + language + '.pth'))
        with open(outpath / ('oscar_' + language + '.dict'), "w") as file1:
            file1.writelines([x + '\t' + str(dict[x]) + '\n' for x in dict])

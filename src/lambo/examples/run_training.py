"""
Script for training LAMBO models using UD data
"""
import sys

import importlib.resources as resources
from pathlib import Path

import torch
from lambo.learning.train import train_new_and_save

if __name__=='__main__':
    treebanks = Path(sys.argv[1]) #Path.home() / 'PATH-TO/ud-treebanks-v2.9/'
    outpath = Path(sys.argv[2]) #Path.home() / 'PATH-TO/models/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Read available languages
    languages_file_str = resources.read_text('lambo.resources', 'languages.txt', encoding='utf-8', errors='strict')
    languages = [line.split(' ')[0] for line in languages_file_str.split('\n') if not line[0] == '#']
    
    for i in range(len(languages)):
        if len(sys.argv)>3 and i % 5 != int(sys.argv[3]):
            continue
        language = languages[i]
        if (outpath / (language + '.pth')).exists():
            continue
        print(str(i) + '/' + str(len(languages)) + '========== ' + language + ' ==========')
        inpath = treebanks / language
        train_new_and_save('LAMBO-BILSTM', inpath, outpath, 20, device)

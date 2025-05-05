"""
Script for training LAMBO models using UD data from pretrained
"""
import sys
from pathlib import Path

import importlib.resources as resources
import torch

from lambo.learning.train import train_new_and_save, train_pretrained_and_save

EPOCHS = 20
SOW_UNKNOWNS = True

if __name__=='__main__':
    treebanks = Path(sys.argv[1]) #Path.home() / 'PATH-TO/ud-treebanks-v2.11/'
    outpath = Path(sys.argv[2]) #Path.home() / 'PATH-TO/models/full/'
    pretrained_path = Path(sys.argv[3]) #Path.home() / 'PATH-TO/models/pretrained/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    languages_file_str = resources.read_text('lambo.resources', 'languages.txt', encoding='utf-8', errors='strict')
    lines = [line.strip() for line in languages_file_str.split('\n') if not line[0] == '#']
    
    for i, line in enumerate(lines):
        if len(sys.argv)>4 and i % 5 != int(sys.argv[4]):
            continue
        parts = line.split()
        model = parts[0]
        language = parts[1]
        #if model != 'UD_Polish-PDB':
        #    continue
        if (outpath / (model + '.pth')).exists():
            continue
        print(str(i) + '/' + str(len(lines)) + '========== ' + model + ' ==========')
        inpath = treebanks / model
        if language != '?':
            train_pretrained_and_save(language, inpath, outpath, pretrained_path, EPOCHS, sow_unknowns=SOW_UNKNOWNS, device = device)
        else:
            train_new_and_save('LAMBO-BILSTM', inpath, outpath, EPOCHS, device = device)

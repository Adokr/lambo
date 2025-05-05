"""
Script for training LAMBO subword splitting models using UD data from pretrained
"""
import time, sys
from pathlib import Path

import importlib.resources as resources
import torch

from lambo.segmenter.lambo import Lambo
from lambo.subwords.train import train_subwords_and_save

if __name__ == '__main__':
    treebanks = Path(sys.argv[1]) #Path.home() / 'PATH-TO/ud-treebanks-v2.11/'
    outpath = Path(sys.argv[2]) #Path.home() / 'PATH-TO/models/full-subwords/'
    segmenting_path = Path(sys.argv[3]) #Path.home() / 'PATH-TO/models/full/'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    languages_file_str = resources.read_text('lambo.resources', 'languages.txt', encoding='utf-8', errors='strict')
    lines = [line.strip() for line in languages_file_str.split('\n') if not line[0] == '#']
    
    start = time.time()
    for i, line in enumerate(lines):
        parts = line.split()
        model = parts[0]
        if (outpath / (model + '_subwords.pth')).exists():
            continue
        print(str(i) + '/' + str(len(lines)) + '========== ' + model + ' ==========')
        inpath = treebanks / model
        segmenter = Lambo.from_path(segmenting_path, model)
        train_subwords_and_save('LAMBO-BILSTM', treebanks / model, outpath, segmenter, epochs=20, device=device)
    print(str(time.time()-start)+' s.')

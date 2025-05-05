"""
Evaluation of available models by comparing to UD gold standard
"""
from pathlib import Path

import importlib.resources as resources

from lambo.evaluation.evaluate import evaluate_segmenter
from lambo.segmenter.lambo import Lambo
from lambo.segmenter.spacy import Spacy_segmenter

if __name__ == '__main__':
    modelpath = Path.home() / 'data/lambo/models/full211-s/'
    modelPpath = Path.home() / 'data/lambo/models/full213-s-withunk/'
    tmp_path = Path.home() / 'data/lambo/out/tmp.conllu'
    
    treebanks = [line.split(' ')[0] for line in
                 resources.read_text('lambo.resources', 'languages.txt', encoding='utf-8', errors='strict').split('\n')
                 if
                 not line[0] == '#']
    
    # Load spaCy segmenter as baseline
    spacy = Spacy_segmenter()
    
    segmenters = {'spaCy': spacy, 'LAMBO': None, 'LAMBO modified': None}
    
    print('Treebank\tMeasure (F1)\t' + '\t'.join(segmenters))
    for treebank in treebanks:
        try:
            # Load LAMBO segmenter
            lambo = Lambo.from_path(modelpath, treebank)
            # Load modified LAMBO segmenter
            lamboP = Lambo.from_path(modelPpath, treebank)
        except FileNotFoundError:
            print("IGNORING "+treebank)
            continue
        
        segmenters['LAMBO'] = lambo
        segmenters['LAMBO modified'] = lamboP
        
        data_path = Path.home() / 'data' / 'lambo' / 'ud-treebanks-v2.11' / treebank
        text_file = list(data_path.glob('*-ud-test.txt'))
        if len(text_file) != 1:
            continue
        text = text_file[0].read_text().replace('\n', ' ')
        
        gold_file = list(data_path.glob('*-ud-test.conllu'))
        if len(gold_file) != 1:
            continue
        gold_file = gold_file[0]
        
        results = {}
        # Evaluate each segmenter
        for segmenter in segmenters:
            results[segmenter] = evaluate_segmenter(segmenters[segmenter], text, gold_file, tmp_path)
        
        # Print the results in TSV format
        categories = ['Tokens', 'Words', 'Sentences']
        for category in categories:
            printout = (treebank if category == categories[0] else '--') + '\t' + category
            bestVal = 0
            bestSeg = ''
            for segmenter in segmenters:
                value = results[segmenter][category]['F1']
                if value > bestVal:
                    bestVal = value
                    bestSeg = segmenter
                printout += ('\t' + format(value, '.4f'))
            printout += '\t' + bestSeg
            print(printout)

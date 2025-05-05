"""
Short demo on using the tuning functionality
"""
from pathlib import Path
from lambo.learning.preprocessing_dict import prepare_dataloaders_withdict
from lambo.learning.train import tune
from lambo.segmenter.lambo import Lambo
from lambo.utils.printer import print_document_to_screen
from lambo.utils.ud_reader import read_document

if __name__=='__main__':
    # Load a desired model
    lambo = Lambo.get('pl')
    
    # This sentence uses ``^`` instead of full stops for sentence separation
    text = "To jest pierwsze zdanie^ Kolejne zdanie kończy się podobnie^ I ostatnie też wygląda tak samo^"
    
    document = lambo.segment(text)
    
    # LAMBO fails to recognise these markers
    print_document_to_screen(document)
    
    # Tuning file suing the same convention, i.e. ``^`` instead of full stop
    tuningpath = Path.home() / 'PATH-TO' / 'tuning.conllu'
    
    # Prepare data
    tuning_doc = read_document(tuningpath, False)
    _, train_dataloader, test_dataloader = prepare_dataloaders_withdict([tuning_doc], [tuning_doc], lambo.model.max_len, 32,
                                                                        False, lambo.dict)
    
    # Tune
    tune(lambo.model, train_dataloader, test_dataloader, 3)
    
    # The new model performs better
    document = lambo.segment(text)
    print_document_to_screen(document)

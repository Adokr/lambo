import torch

from lambo.learning.train import tune
from lambo.subwords.model import LamboSubwordNetwork
from lambo.subwords.preprocessing import prepare_subwords_dataloaders
from lambo.utils.ud_reader import read_treebank


def train_subwords_and_save(model_name, treebank_path, save_path, lambo_segmenter, epochs=20, device=torch.device('cpu')):
    """
    Train a new LAMBO subwords model and save it in filesystem.

    :param model_name: type of model trained, currently only ``LAMBO-BILSTM`` is supported
    :param treebank_path: path to the treebank training data
    :param save_path: path to save the generated model
    :param lambo_segmenter: LAMBO segmenter to base on
    :param epochs: number of epochs to run for (default: 20)
    :param device: the device to use for computation
    :return: no value returned
    """
    if model_name not in ['LAMBO-BILSTM']:
        print(" Unrecognised model name: " + model_name)
        return
    
    print("Reading data.")
    train_doc, dev_doc, test_doc = read_treebank(treebank_path, True)
    
    print("Preparing data")
    BATCH_SIZE = 32
    print("Initiating the model.")
    
    MAX_LEN = 32
    train_dataloader, test_dataloader = prepare_subwords_dataloaders([train_doc, dev_doc], [test_doc],
                                                                     MAX_LEN,
                                                                     BATCH_SIZE, lambo_segmenter.dict)
    if train_dataloader is None:
        print("Not enough data to train, moving on.")
        return
    
    model = LamboSubwordNetwork(MAX_LEN, lambo_segmenter.dict, lambo_segmenter.model)
    
    tune(model, train_dataloader, test_dataloader, epochs, device)
    
    print("Saving")
    torch.save(model, save_path / (treebank_path.name + '_subwords.pth'))

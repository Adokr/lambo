"""
A collection of functions useful for training models.
"""
import torch
from torch.optim import Adam

from lambo.learning.model import LamboNetwork
from lambo.learning.preprocessing_dict import utf_category_dictionary, prepare_dataloaders_withdict
from lambo.segmenter.lambo import Lambo
from lambo.utils.ud_reader import read_treebank


def train_loop(dataloader, model, optimizer, device=torch.device('cpu')):
    """
    Training loop.
    
    :param dataloader: dataloader with training data
    :param model: model to be optimised
    :param optimizer: optimiser used
    :param device: the device to use for computation
    :return: no value returned
    """
    size = len(dataloader.dataset)
    for batch, XY in enumerate(dataloader):
        XY = [xy.to(device) for xy in XY]
        Xs = XY[:-1]
        Y = XY[-1]
        pred = model(*Xs)
        loss = model.compute_loss(pred, Y, Xs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(Xs[0])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, device=torch.device('cpu')):
    """
    Test loop.
    
    :param dataloader: dataloader with test data
    :param model: model to be tested
    :param device: the device to use for computation
    :return: no value returned
    """
    num_batches = len(dataloader)
    test_loss = 0
    correct = [0, 0, 0, 0]
    size = [0, 0, 0, 0]
    with torch.no_grad():
        for XY in dataloader:
            XY = [xy.to(device) for xy in XY]
            Xs = XY[:-1]
            Y = XY[-1]
            pred = model(*Xs)
            test_loss += model.compute_loss(pred, Y, Xs).item()
            if len(pred.shape) == 4:
                # Predicting character types (segmentation)
                for i in range(pred.shape[2]):
                    A = pred[:, :, i, :].argmax(2)
                    B = Y[:, :, i]
                    nontrivial = torch.nonzero(A + B, as_tuple=True)
                    equals = (A == B)[nontrivial].type(torch.float)
                    correct[i] += equals.sum().item()
                    size[i] += torch.numel(equals)
            elif len(pred.shape) == 3:
                # Predictiong characters (subword prediction)
                A = pred.argmax(2)
                B = Y
                nontrivial = torch.nonzero(Y, as_tuple=True)
                equals = (A == B)[nontrivial].type(torch.float)
                # equals = (A==B).type(torch.float)
                correct[0] += equals.sum().item()
                size[0] += torch.numel(equals)
                pass
    
    test_loss /= num_batches
    size = [s if s > 0 else 1 for s in size]
    print(
        f"Test Error: \n Accuracy chars: {(100 * (correct[0] / size[0])):>5f}%, tokens: {(100 * (correct[1] / size[1])):>5f}%, mwtokens: {(100 * (correct[2] / size[2])):>5f}%, sentences: {(100 * (correct[3] / size[3])):>5f}%, Avg loss: {test_loss:>8f} \n")


def test_loop_pretraining(dataloader, model, device=torch.device('cpu')):
    """
    Test loop for pretraining.

    :param dataloader: dataloader with test data
    :param model: model to be tested
    :param device: the device to use for computation
    :return: no value returned
    """
    num_batches = len(dataloader)
    test_loss = 0
    correct = [0, 0]
    size = [0, 0]
    with torch.no_grad():
        for XY in dataloader:
            XY = [xy.to(device) for xy in XY]
            Xs = XY[:-1]
            Y = XY[-1]
            pred = model(*Xs)
            test_loss += model.compute_loss(pred, Y, Xs).item()
            nontrivial = torch.nonzero(Xs[-1], as_tuple=True)
            equals = (Y == pred.argmax(-1))[nontrivial].type(torch.float)
            correct[0] += equals.sum().item()
            size[0] += torch.numel(equals)
            equals = (Y == pred.argmax(-1)).type(torch.float)
            correct[1] += equals.sum().item()
            size[1] += torch.numel(equals)
    test_loss /= num_batches
    print(
        f"Test Error: \n Accuracy nontrivial: {(100 * (correct[0] / size[0])):>5f}%, trivial: {(100 * (correct[1] / size[1])):>5f}%, Avg loss: {test_loss:>8f} \n")


def train_new_and_save(model_name, treebank_path, save_path, epochs, sow_unknowns=False, device=torch.device('cpu')):
    """
    Train a new LAMBO model and save it in filesystem.
    
    :param model_name: type of model trained, currently only ``LAMBO-BILSTM`` is supported
    :param treebank_path: path to the treebank training data
    :param save_path: path to save the generated model
    :param epochs: number of epochs to run for
    :param sow_unknowns: whether to randomly mask some characters as unknown
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
    
    MAX_LEN = 256
    dict, train_dataloader, test_dataloader = prepare_dataloaders_withdict([train_doc, dev_doc], [test_doc],
                                                                           MAX_LEN,
                                                                           BATCH_SIZE, sow_unknowns)
    model = LamboNetwork(MAX_LEN, dict, len(utf_category_dictionary))
    
    tune(model, train_dataloader, test_dataloader, epochs, device)
    
    print("Saving")
    torch.save(model, save_path / (treebank_path.name + '.pth'))
    with open(save_path / (treebank_path.name + '.dict'), "w") as file1:
        file1.writelines([x + '\t' + str(dict[x]) + '\n' for x in dict])


def train_pretrained_and_save(language, treebank_path, save_path, pretrained_path, epochs, sow_unknowns=False,
                              device=torch.device('cpu')):
    """
    Train a new LAMBO model, staring from pretrained, and save it in filesystem.

    :param language: ISO code of the language to use as pretraining
    :param treebank_path: path to the treebank training data
    :param save_path: path to save the generated model
    :param pretrained_path: path to the pretraining models
    :param epochs: number of epochs to run for
    :param sow_unknowns: whether to randomly mask some characters as unknown
    :param device: the device to use for computation
    :return: no value returned
    """
    print("Loading pretrained model")
    pretrained_name = 'oscar_' + language
    file_path = pretrained_path / (pretrained_name + '.pth')
    if not file_path.exists():
        print("Pretrained model not found, falling back to training from scratch.")
        return train_new_and_save('LAMBO-BILSTM', treebank_path, save_path, epochs, device)
    pretrained_model = torch.load(file_path, map_location=torch.device('cpu'))
    dict = Lambo.read_dict(pretrained_path / (pretrained_name + '.dict'))
    
    print("Reading data.")
    train_doc, dev_doc, test_doc = read_treebank(treebank_path, True)
    
    print("Initiating the model.")
    MAX_LEN = 256
    model = LamboNetwork(MAX_LEN, dict, len(utf_category_dictionary), pretrained=pretrained_model)
    
    print("Preparing data")
    BATCH_SIZE = 32
    
    dict, train_dataloader, test_dataloader = prepare_dataloaders_withdict([train_doc, dev_doc], [test_doc],
                                                                           MAX_LEN,
                                                                           BATCH_SIZE, sow_unknowns, dict=dict)
    
    tune(model, train_dataloader, test_dataloader, epochs, device)
    
    print("Saving")
    torch.save(model, save_path / (treebank_path.name + '.pth'))
    with open(save_path / (treebank_path.name + '.dict'), "w") as file1:
        file1.writelines([x + '\t' + str(dict[x]) + '\n' for x in dict])


def tune(model, train_dataloader, test_dataloader, epochs, device=torch.device('cpu')):
    """
    Tune an existing LAMBO model with the provided data
    
    :param model: model to be tuned
    :param train_dataloader: dataloader for training data
    :param test_dataloader: dataloader for test data
    :param epochs: number of epochs to run for
    :param device: the device to use for computation
    :return: no value returned
    """
    print("Preparing training")
    model.to(device)
    learning_rate = 1e-3
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    print("Training")
    test_loop(test_dataloader, model, device)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer, device)
        test_loop(test_dataloader, model, device)


def pretrain(model, train_dataloader, test_dataloader, epochs, device=torch.device('cpu')):
    """
    Tune an existing LAMBO pretraining model with the provided data
    
    :param model: model to be tuned
    :param train_dataloader: dataloader for training data
    :param test_dataloader: dataloader for test data
    :param epochs: number of epochs to run for
    :param device: the device to use for computation
    :return: no value returned
    """
    print("Preparing pretraining")
    model.to(device)
    learning_rate = 1e-3
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    print("Pretraining")
    test_loop_pretraining(test_dataloader, model, device)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer, device)
        test_loop_pretraining(test_dataloader, model, device)

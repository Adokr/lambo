import torch
from torch.nn import Module, Embedding, LSTM, Linear, LogSoftmax, NLLLoss


class LamboPretrainingNetwork(Module):
    """
    LAMBO neural pretraining model. Ananlogous to ``LamboNetwork``, but predicts the masked characters.
    """
    
    def __init__(self, max_len, dict, utf_categories_num):
        """
        Create a LAMBO pretraining neural network.
        
        :param max_len: maximum length of an input sequence,
        :param dict: character dictionary
        :param utf_categories_num: number of UTF categories
        """
        super(LamboPretrainingNetwork, self).__init__()
        self.max_len = max_len
        self.embedding_layer = Embedding(len(dict), 64, dict['<PAD>'])
        self.lstm_layer = LSTM(input_size=self.embedding_layer.embedding_dim + utf_categories_num, hidden_size=256,
                               batch_first=True,
                               bidirectional=True)
        self.linear_layer = Linear(self.lstm_layer.hidden_size * 2, len(dict))
        self.softmax_layer = LogSoftmax(2)
        self.loss_fn = NLLLoss()
    
    def forward(self, x_char, x_utf, x_mask):
        """
        Computation of the network output.
        
        :param x_char: a tensor of BxL character indices,
        :param x_utf: a tensor of BxLxU UTF category indicators,
        :param x_mask: (unused)
        :return: a tensor of BxLxD class scores
        
        Where B = batch size, L = maximum sequence length, D = character dictionary size
        """
        embedded = self.embedding_layer(x_char)
        with_category = torch.cat((embedded, x_utf), 2)
        hidden = self.lstm_layer(with_category)[0]
        scores = self.linear_layer(hidden)
        probabilities = self.softmax_layer(scores)
        return probabilities
    
    def compute_loss(self, pred, true, Xs):
        """
        Comput cross-entropy loss.

        :param pred: tensor with predicted class probabilities
        :param true: tensor with true classes (character IDs)
        :param Xs: network input, used for ignoring non-masked characters
        :return: loss value
        """
        pred = torch.reshape(pred, (-1, pred.shape[-1]))
        true = torch.reshape(true, (-1,))
        X_mask = torch.reshape(Xs[-1], (-1,))
        nontrivial = torch.nonzero(X_mask, as_tuple=True)
        pred_nontrivial = pred[nontrivial]
        true_nontrivial = true[nontrivial]
        output = self.loss_fn(pred_nontrivial, true_nontrivial)
        return output

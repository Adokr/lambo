import copy,torch

from torch.nn import Module, Embedding, LSTM, Linear, LogSoftmax, NLLLoss


class LamboNetwork(Module):
    """
    LAMBO neural network model. The network has four layers:
    
    * embedding layers for characters, representing each as a 32-long vector (or 64-long),
    * bidirectional LSTM layer, taking a concatenation of (1) character embedding and (2) one-hot UTF category vector as input and outputting 2*128-long state vector (or 2*256),
    * dense linear layer, converting LSTM state vectors to class scores
    * softmax layer, computing probability of eight events for any character:
    
        * is it part of a token (or whitespace)
        * is it the last character of a token (or not)
        * is it the last character of a multi-word token (or not)
        * is it the last character of a sentence (or not)
    """
    
    def __init__(self, max_len, dict, utf_categories_num, pretrained=None):
        """
        Create a LAMBO neural network.
        
        :param max_len: maximum length of an input sequence,
        :param dict: character dictionary
        :param utf_categories_num: number of UTF categories
        :param pretrained: either ``None`` (for new models) or an instance of ``LamboPretrainingNetwork`` (if using pretraining data)
        """
        super(LamboNetwork, self).__init__()
        self.max_len = max_len
        if pretrained is not None:
            # Copy the weights of the embedding and LSTM layers of pretraining model
            self.embedding_layer = copy.deepcopy(pretrained.embedding_layer)
            self.lstm_layer = copy.deepcopy(pretrained.lstm_layer)
        else:
            self.embedding_layer = Embedding(len(dict), 64, dict['<PAD>'])
            self.lstm_layer = LSTM(input_size=self.embedding_layer.embedding_dim + utf_categories_num, hidden_size=256,
                                   batch_first=True,
                                   bidirectional=True)
        self.linear_layer = Linear(self.lstm_layer.hidden_size * 2, 8)
        self.softmax_layer = LogSoftmax(3)
        self.loss_fn = NLLLoss()
    
    def forward(self, x_char, x_utf):
        """
        Computation of the network output.
        
        :param x_char: a tensor of BxL character indices,
        :param x_utf: a tensor of BxLxU UTF category indicators,
        :return: a tensor of BxLx4x2 class scores
        
        Where B = batch size, L = maximum sequence length, U = number of UTF categories
        """
        embedded = self.embedding_layer(x_char)
        with_category = torch.cat((embedded, x_utf), 2)
        hidden = self.lstm_layer(with_category)[0]
        scores = self.linear_layer(hidden)
        predictions = torch.reshape(scores, (-1, self.max_len, 4, 2))
        probabilities = self.softmax_layer(predictions)
        return probabilities
    
    def compute_loss(self, pred, true, Xs):
        """
        Comput cross-entropy loss.
        
        :param pred: tensor with predicted class probabilities
        :param true: tensor witrh true classes
        :param Xs: (not used)
        :return: loss value
        """
        pred = torch.reshape(pred, (-1, 2))
        true = torch.reshape(true, (-1,))
        output = self.loss_fn(pred, true)
        return output
    
    @staticmethod
    def postprocessing(Y, text):
        """
        Postprocessing the model output by reshaping and trimming it.
        
        :param Y: model output
        :param text: text for which the predictions are made
        :return: fixed output
        """
        decisions = torch.reshape(Y.argmax(3), (-1, 4)).numpy()
        decisions = decisions[:len(text)]
        return decisions

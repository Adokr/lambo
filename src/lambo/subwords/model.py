import torch
from torch.nn import Embedding, LSTM, Linear, LogSoftmax, NLLLoss, Module


class LamboSubwordNetwork(Module):
    """
    LAMBO subword splitting neural network model. The network has four layers:

    * embedding layers for characters, representing each as a 64-long vector,
    * bidirectional LSTM layer, taking a character embedding as input and outputting 2*64-long state vector,
    * dense linear layer, converting LSTM state vectors to 64-dimensional embedding space
    * inverted embedding layer to convert back to characters using the same matrix as for embedding
    * softmax layer, computing probability of any character
    """
    
    def __init__(self, max_len, dict, pretrained=None):
        """
        Create a LAMBO subword neural network.

        :param max_len: maximum length of an input sequence (i.e. characters in a token),
        :param dict: character dictionary
        :param pretrained: either ``None`` (for virgin models) or an instance of ``LamboNetwork`` (if using pretraining data)
        """
        super(LamboSubwordNetwork, self).__init__()
        self.max_len = max_len
        self.dict = dict
        if pretrained is not None:
            # Copy the weights of the embedding of pretraining model
            self.embedding_layer = Embedding.from_pretrained(pretrained.embedding_layer.weight, freeze=False,
                                                             padding_idx=None)
        else:
            self.embedding_layer = Embedding(len(dict), 64, dict['<PAD>'])
        self.lstm_layer = LSTM(input_size=self.embedding_layer.embedding_dim, hidden_size=64, batch_first=True,
                               bidirectional=True)
        self.linear_layer = Linear(self.lstm_layer.hidden_size * 2, self.embedding_layer.embedding_dim)
        self.softmax_layer = LogSoftmax(2)
        self.loss_fn = NLLLoss()
    
    def forward(self, x_char):
        """
        Computation of the network output (B = batch size, L = maximum sequence length, V = number of words in the dictionary)

        :param x_char: a tensor of BxL character indices,
        :return: a tensor of BxLxV class scores
        """
        embedded = self.embedding_layer(x_char)
        hidden = self.lstm_layer(embedded)[0]
        reduced = self.linear_layer(hidden)
        
        # Computing inverted embedding as a cosine similarity score of the transformed representation and original embeddings
        scores = self.inverted_embedding(reduced, self.embedding_layer)
        
        probabilities = self.softmax_layer(scores)
        return probabilities
    
    @staticmethod
    def inverted_embedding(input, embedding_layer):
        """
        Inverted embeddings matrix. Finds the best items (i.e. characters or words) in the dictionary of the
        original embedding layer (B = batch size, L = maximum sequence length, E = embedding size, V = number of words
         in the dictionary) for the input in the embedding space.
        
        :param input: a tensor in the hidden space of shape BxLxE
        :param embedding_layer: an embedding layer with VxE parameter matrix
        :return: dot product similarity tensor of shape BxLxV
        """
        # Normalise both matrices
        input_normalised = torch.nn.functional.normalize(input, dim=2)
        weights_normalised = torch.nn.functional.normalize(embedding_layer.weight.data, dim=1)
        # Dot product of normalised vectors equals cosine similarity
        scores = torch.matmul(input_normalised, torch.transpose(weights_normalised, 0, 1))
        return scores
    
    def compute_loss(self, pred, true, Xs):
        """
        Comput cross-entropy loss.

        :param pred: tensor with predicted character probabilities
        :param true: tensor witrh true classes
        :param Xs: (not used)
        :return: loss value
        """
        pred = torch.reshape(pred, (-1, len(self.dict)))
        true = torch.reshape(true, (-1,))
        output = self.loss_fn(pred, true)
        return output

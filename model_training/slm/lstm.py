import torch
import torch.nn as nn
from slm.utils import sinusoidal_positional_encoding


class LSTM(nn.Module):
    """
    Simple implementation of an LSTM based on the description in
    https://colah.github.io/posts/2015-08-Understanding-LSTMs/.

    parameters:
        vocab_size: the size of the vocabulary as an integer
        embedding_dim: the size of the embedding space
    """

    def __init__(self, vocab_size, embedding_dim):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size

        self.forget_gate = nn.Linear(embedding_dim * 2, embedding_dim)
        self.input_gate = nn.Linear(embedding_dim * 2, embedding_dim)
        self.tanh_layer = nn.Linear(embedding_dim * 2, embedding_dim)
        self.output_filter = nn.Linear(embedding_dim * 2, embedding_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h, c):
        """
        The forward pass through the LSTM.

        parameters:
            x: the embedded word at time step t
            h: the output of the previous block at time t-1
            c: the encoded memory or context

        outputs:
            hnew: the output vector that is then put through an mlp to get
                next word predictions
            cnew: the updated memory or context vector

        *initial h and c should be 0s... effectively a
        seed from which to build memory.
        """

        input = torch.cat([x, h], dim=1)
        f = self.sigmoid(self.forget_gate(input))
        i = self.sigmoid(self.input_gate(input))
        ctilda = self.tanh(self.tanh_layer(input))
        cnew = f * c + i * ctilda

        output_filter = self.sigmoid(self.output_filter(input))
        hnew = output_filter * self.tanh(cnew)

        return hnew, cnew


class Embedding(nn.Module):
    """
    Embedding of codified words with positional encoding.

    parameters:
        embedding_dim: the size of the embedding space
        vocab_size: the size of the vocbulary
        context_window_size: longest number of tokens that will be passed to the
            lstm during training.
    """

    def __init__(
        self, embedding_dim: int, vocab_size: int, context_window_size: int
    ):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size + 1, embedding_dim, padding_idx=0
        )

        self.pos_enc = sinusoidal_positional_encoding(
            context_window_size, embedding_dim
        )

    def forward(self, x):
        """
        Forward pass of the embedding.

        parameters:
            x: codified token

        outputs:
            embedding: embedded token with positional encoding
        """
        print(x.shape)
        embed = self.embedding(x) + self.pos_enc[: x.size(1)].to(x.device)
        return embed


class MLP(nn.Module):
    """
    MLP to translate the output of the LSTM into a next word prediction.

    parameters:
        embedding_dim: size of the embedding space
        mlp_layers: number of layers in the mlp
        mlp_dim: number of hidden nodes in each layer
        vocab_size: the size of the vocabulary
    """

    def __init__(self, embedding_dim, mlp_layers, mlp_dim, vocab_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(embedding_dim, mlp_dim))
        for _ in range(mlp_layers):
            self.layers.append(nn.Linear(mlp_dim, mlp_dim))
        self.layers.append(nn.Linear(mlp_dim, vocab_size))

    def forward(self, x):
        """
        Forward pass of the MLP.

        parameters:
            x: the output from the LSTM

        output:
            x: probability over vocabulary for the next token in the sequence
        """

        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

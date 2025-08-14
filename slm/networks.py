import torch.nn as nn
import torch


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, mlp_layers, mlp_dim):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.layers = nn.ModuleList()
        for _ in range(mlp_layers):
            self.layers.append(nn.Linear(embedding_dim, mlp_dim))
            self.layers.append(nn.Linear(mlp_dim, embedding_dim))
        

    def forward(self, x):
        x = self.embedding(x)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        # mask self-attention
        # Create a mask to allow each token to attend only to itself and future tokens (causal mask)
        mask = torch.triu(torch.ones_like(attention_scores, dtype=torch.bool), diagonal=0)
        attention_scores = torch.where(mask, attention_scores, float('-inf'))
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        x = torch.matmul(attention_weights, value)
        x = torch.nn.functional.relu(x)  # Apply activation function

        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.relu(x)

        return x
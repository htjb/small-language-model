import pickle

import numpy as np
import torch
import yaml
from slm.byte_pair_encoding import bpe  # Import the bpe class
from slm.networks import Transformer  # Import the Embedding class

np.random.seed(42)
torch.manual_seed(42)


def step(
    vector: torch.Tensor,
    transform: Transformer,
    criterion: torch.nn.CrossEntropyLoss,
):
    output = transform(vector[0].unsqueeze(0))
    target = torch.tensor(vector[0][1:])
    loss = criterion(output[0, :-1], target)
    return loss, output, target


hyperparameters = yaml.safe_load(
    open("classic_books_hyperparameters.yaml", "r")
)

vocab_model = pickle.load(open("classic_books_vocab.pkl", "rb"))

transform = Transformer(
    vocab_size=len(vocab_model.word_to_index) + 1,
    embedding_dim=hyperparameters["embedding_size"],
    mlp_layers=hyperparameters["mlp_layers"],
    mlp_dim=hyperparameters["mlp_dim"],
    context_window_size=hyperparameters["context_window_size"],
    nheads=hyperparameters["nheads"],
    predict=True,
    entropy=hyperparameters["entropy"],
)  # Create an instance of the Transformer class

state_dict = torch.load(
    "classic_books_model.pth", map_location=torch.device("cpu")
)  # Load the state dictionary

transform.load_state_dict(
    state_dict
)  # Load the state dictionary into the model

test_phrase = "what is the weather "  # Define a test phrase
# only need to make pass through the mlp for the last word... will need to think
# about how to do this in the future

vector = vocab_model.codify(test_phrase)  # [:-1]

print(vector)

while (
    vector[-1] != vocab_model.word_to_index["EOS"]
    and len(vector) < hyperparameters["context_window_size"]
):
    output = transform(vector.unsqueeze(0))
    output[0, -1, 0] = -float("Inf")  # zero out PAD class

    probs = (
        torch.nn.functional.softmax(output["output"][0, -1, :], dim=0)
        .detach()
        .numpy()
    )

    probs[int(vector[-1])] =0  # zero out the previous word

    # optional: top-k sampling
    k = 250
    top_k_indices = probs.argsort()[-k:]
    top_k_probs = probs[top_k_indices]
    top_k_probs /= top_k_probs.sum()  # normalize
    out = np.random.choice(top_k_indices, p=top_k_probs)

    vector = torch.cat(
        (vector, torch.tensor([int(out)])), dim=0
    )  # Append the predicted word to the vector

print([vocab_model.index_to_word[int(i)] for i in vector])
if type(vocab_model) is bpe:
    print(
        "".join([vocab_model.index_to_word[int(i)] for i in vector])
    )  # Print the result
else:
    print(" ".join([vocab_model.index_to_word[int(i)] for i in vector]))

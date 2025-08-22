import numpy as np
import torch
import yaml
from slm.bag_of_words import bag_of_words  # Import the bag_of_words class
from slm.networks import Transformer  # Import the Embedding class


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
    open("alice_in_wonderland_hyperparameters.yaml", "r")
)

bow = bag_of_words()

transform = Transformer(
    vocab_size=len(bow.word_to_index) + 1,
    embedding_dim=hyperparameters["embedding_size"],
    mlp_layers=hyperparameters["mlp_layers"],
    mlp_dim=hyperparameters["mlp_dim"],
    context_window_size=hyperparameters["context_window_size"],
    nheads=hyperparameters["nheads"],
    predict=True,
    entropy=hyperparameters["entropy"],
)  # Create an instance of the Transformer class

state_dict = torch.load(
    "alice_in_wonderland_model.pth", map_location=torch.device("cpu")
)  # Load the state dictionary

transform.load_state_dict(
    state_dict
)  # Load the state dictionary into the model

test_phrase = "Alice was beginning"  # Define a test phrase
# only need to make pass through the mlp for the last word... will need to think
# about how to do this in the future

index_to_word = {i: w for w, i in bow.word_to_index.items()}
vector = bow.codify(test_phrase)
while vector[-1] != bow.word_to_index["EOS"]:
    output = transform(vector.unsqueeze(0))
    out = np.argmax(output["output"][0, -1, 1:].detach().numpy())
    predicted_word = index_to_word[int(out)]
    vector = torch.cat(
        (vector, torch.tensor([out])), dim=0
    )  # Append the predicted word to the vector

print(
    " ".join([index_to_word[int(i)] for i in vector[:-1]])
)  # Print the result

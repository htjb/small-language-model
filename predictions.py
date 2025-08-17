from slm.bag_of_words import bag_of_words  # Import the bag_of_words class
from slm.networks import Transformer  # Import the Embedding class
import torch
import numpy as np
import yaml

def step(vector: torch.Tensor, transform: Transformer, 
         criterion: torch.nn.CrossEntropyLoss):
    output = transform(vector[0].unsqueeze(0))
    target = torch.tensor(vector[0][1:])
    loss = criterion(output[0, :-1], target)
    return loss, output, target


hyperparameters = yaml.safe_load(open('alice_in_wonderland_hyperparameters.yaml', 'r')) 
batch_size = hyperparameters['batch_size']  # Define the batch size
embedding_size = hyperparameters['embedding_size']  # Define the embedding size
mlp_layers = hyperparameters['mlp_layers']  # Define the number of MLP layers
mlp_dim = hyperparameters['mlp_dim']  # Define the MLP dimension
context_window_size = hyperparameters['context_window_size']  # Define the context window size

bow = bag_of_words() 
line = "Alice was beginning"
vector = bow.codify(line)  

transform = Transformer(vocab_size=len(bow.word_to_index), 
                embedding_dim=embedding_size, mlp_layers=mlp_layers, mlp_dim=mlp_dim,
                context_window_size=context_window_size)  # Create an instance of the Transformer class

state_dict = torch.load('alice_in_wonderland_model.pth', map_location=torch.device('cpu'))  # Load the state dictionary

transform.load_state_dict(state_dict)  # Load the state dictionary into the model

test_phrase = "how are you"  # Define a test phrase
# only need to make pass through the mlp for the last word... will need to think
# about how to do this in the future
output = transform(bow.codify(test_phrase).unsqueeze(0))

output = torch.softmax(output, dim=-1)  # Apply softmax to the output
print(output)
print("Output shape:", output.shape)  # Print the shape of the output
# the last ouput is the prediction for the next word
for j in range(len(test_phrase.split(" "))):
    out = np.argmax(output[0, j, :].detach().numpy())
    index_to_word = {i: w for w, i in bow.word_to_index.items()}
    predicted_word = index_to_word[out]
    print("Predicted words:", predicted_word)  # Print the predicted words
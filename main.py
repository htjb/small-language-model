from slm.bag_of_words import bag_of_words  # Import the bag_of_words class
from slm.networks import Transformer  # Import the Embedding class
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting data
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler  # Import Dataset and DataLoader for handling data
import torch.optim as optim
from tqdm import tqdm  # Import tqdm for progress bar
import torch
import numpy as np
import yaml

def step(vector: torch.Tensor, transform: Transformer, 
         criterion: torch.nn.CrossEntropyLoss):
    output = transform(vector[0][:-1].unsqueeze(0))
    target = torch.tensor(vector[0][1:])
    loss = criterion(output[0], target)
    return loss, output, target


batch_size = 32 # Define the batch size
embedding_size = 128  # Define the embedding size
mlp_layers = 5  # Define the number of MLP layers
mlp_dim = 512  # Define the MLP dimension
context_window_size = 512  # Define the context window size
nheads = 2

hyperparameters = {
    'embedding_size': embedding_size,
    'mlp_layers': mlp_layers,
    'mlp_dim': mlp_dim,
    'context_window_size': context_window_size,
    'batch_size': batch_size,
    'nheads': nheads  # Define the number of attention heads
}

bow = bag_of_words()

transform = Transformer(vocab_size=len(bow.word_to_index), 
                embedding_dim=embedding_size, 
                mlp_layers=mlp_layers, mlp_dim=mlp_dim,
                context_window_size=context_window_size,
                nheads=nheads)  # Create an instance of the Transformer class

with open('alice-in-wonderland.txt', 'r') as file:
    text = file.readlines()  # Read the text file line by line 


codified_text = [bow.codify(t) for t in text if t.strip()]  # Codify each line of the text
codified_text = torch.concat(codified_text, dim=0)  # Concatenate the codified lines into a single tensor

train, test = train_test_split(codified_text, test_size=0.2, random_state=42)
train_text = torch.tensor(train, dtype=torch.long)
test = torch.tensor(test, dtype=torch.long)
test, val = train_test_split(test, test_size=0.5, random_state=42)
test_text = torch.tensor(test, dtype=torch.long) 
val_text = torch.tensor(val, dtype=torch.long)

train_text = TensorDataset(train_text) 
test_text = TensorDataset(test_text)
val_text = TensorDataset(val_text)

# Create all indices for complete batches
num_complete_batches = len(train_text) // batch_size
total_indices = num_complete_batches * batch_size

# Create batch-wise shuffled indices
indices = np.arange(total_indices).reshape(-1, batch_size)
np.random.shuffle(indices)  # Shuffle rows (batches)
shuffled_indices = indices.flatten()

sampler = SubsetRandomSampler(shuffled_indices)

train_dataloader = DataLoader(train_text, batch_size=batch_size, sampler=sampler, shuffle=False)
test_dataloader = DataLoader(test_text, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_text, batch_size=batch_size, shuffle=False)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(transform.parameters(), lr=1e-4, weight_decay=1e-5)  # Use AdamW optimizer

best_loss = float('inf')  # Initialize best loss
best_model = None  # Placeholder for the best model
patience_counter = 0  # Initialize patience counter
patience = 50

pbar = tqdm(range(1000), desc="Training Progress")  # Initialize progress bar

for epoch in pbar:  # Number of epochs
    optimizer.zero_grad()
    total_loss = 0.0
    for vector in train_dataloader:
        loss, _, _ = step(vector, transform, criterion)  # Perform a training step
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_dataloader)  # Average loss for the epoch

    val_loss = 0.0
    with torch.no_grad():
        for vector in val_dataloader:
            loss, _, _ = step(vector, transform, criterion)  # Perform a validation step
            val_loss += loss.item()
        val_loss = val_loss / len(val_dataloader)  # Average validation loss
    
    if val_loss < best_loss:
        best_loss = val_loss
        best_model = transform.state_dict()  # Save the best model
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}, " + 
              f"best validation loss: {best_loss}")
        break

    pbar.set_postfix({
        'avg_loss': avg_loss,
        'val_loss': val_loss,
        'best_loss': best_loss,
        'patience_counter': patience_counter
    })  # Update progress bar with current losses

transform.load_state_dict(best_model)  # Load the best model

torch.save(transform.state_dict(), 'alice_in_wonderland_model.pth')
with open('alice_in_wonderland_hyperparameters.yaml', 'w') as f:
    yaml.dump(hyperparameters, f)  # Save hyperparameters to a YAML file

transform.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_loss = 0.0
    correct, incorrect = 0, 0
    for vector in test_dataloader:
        loss, output, target = step(vector, transform, criterion)
        pred = torch.argmax(output[0], dim=1)
        correct += (pred == target).sum().item()
        incorrect += (pred != target).sum().item()
        test_loss += loss.item()
    test_loss /= len(test_dataloader)  # Average test loss
    print(f"Test Loss: {test_loss}")  # Print the test loss
    print(f"Accuracy: {correct / (correct + incorrect) * 100:.2f}%")  # Print the accuracy
    print(f"Correct: {correct}, Incorrect: {incorrect}")  # Print the number of correct and incorrect predictions
print("Vocabulary size:", len(bow.word_to_index))  # Print the vocabulary size


# only need to make pass through the mlp for the last word... will need to think
# about how to do this in the future
output = transform(torch.tensor(bow.codify("Alice was beginning")).unsqueeze(0))
print("Output shape:", output.shape)  # Print the shape of the output
# the last ouput is the prediction for the next word
output = np.argmax(output[0, -1, :].detach().numpy())
index_to_word = {i: w for w, i in bow.word_to_index.items()}
predicted_word = index_to_word[output]
print("Predicted words:", predicted_word)  # Print the predicted words
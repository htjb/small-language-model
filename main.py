from slm.bag_of_words import bag_of_words  # Import the bag_of_words class
from slm.networks import Transformer  # Import the Embedding class
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting data
from torch.utils.data import TensorDataset, DataLoader  # Import Dataset and DataLoader for handling data
import torch.optim as optim
import torch
import numpy as np

batch_size = 64  # Define the batch size

bow = bag_of_words() 
line = "Alice was beginning"
vector = bow.codify(line)  

transform = Transformer(vocab_size=len(bow.word_to_index), 
                embedding_dim=64, mlp_layers=1, mlp_dim=256,
                context_window_size=100)  # Create an instance of the Transformer class

with open('alice-in-wonderland.txt', 'r') as file:
    text = file.readlines()  # Read the text file line by line 


codified_text = [bow.codify(t) for t in text if t.strip()]  # Codify each line of the text
codified_text = torch.concat(codified_text, dim=0)  # Concatenate the codified lines into a single tensor

train, test = train_test_split(codified_text, test_size=0.3, random_state=42)
train_text = torch.tensor(train, dtype=torch.long)
test = torch.tensor(test, dtype=torch.long)
test, val = train_test_split(test, test_size=0.5, random_state=42)
test_text = torch.tensor(test, dtype=torch.long) 
val_text = torch.tensor(val, dtype=torch.long)

train_text = TensorDataset(train_text) 
test_text = TensorDataset(test_text)
val_text = TensorDataset(val_text)

train_dataloader = DataLoader(train_text, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_text, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_text, batch_size=batch_size, shuffle=False)

"""train_batches = next(iter(train_dataloader))
train_batches_perm = torch.randperm(len(train_batches))
train_batches = train_batches[train_batches_perm]  # Shuffle the training batches
print(train_batches.shape)  # Print the shape of the training batches
exit()"""

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(transform.parameters(), lr=1e-4)  # Use AdamW optimizer

best_loss = float('inf')  # Initialize best loss
patience_counter = 0  # Initialize patience counter
patience = 25

for epoch in range(100):  # Number of epochs
    optimizer.zero_grad()
    total_loss = 0.0
    count = 0
    for vector in train_batches:
        output = transform(vector[0].unsqueeze(0))  # Add batch dimension
        target = torch.tensor(vector[0][1:])
        loss = criterion(output[0, :-1], target)
        total_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / count if count > 0 else 0

    val_loss = 0.0
    val_count = 0
    with torch.no_grad():
        for vector in val_dataloader:
            output = transform(vector[0].unsqueeze(0))
            target = torch.tensor(vector[0][1:])
            loss = criterion(output[0, :-1], target)
            val_loss += loss.item()
            val_count += 1
        
        val_loss = val_loss / val_count if val_count > 0 else 0
    
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}, best validation loss: {best_loss}")
        break
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss}, Validation Loss: {val_loss}, Best Loss: {best_loss}, Counter: {patience_counter}")

transform.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_loss = 0.0
    count = 0
    for vector in test_dataloader:
        output = transform(vector[0].unsqueeze(0))
        target = torch.tensor(vector[0][1:])
        loss = criterion(output[0, :-1], target)
        test_loss += loss.item()
        count += 1
    test_loss /= count if count > 0 else 0
    print(f"Test Loss: {test_loss}")  # Print the test loss

output = transform(torch.tensor(bow.codify("how are you")).unsqueeze(0))
print("Output shape:", output.shape)  # Print the shape of the output
output = np.argmax(output[0, -1, :].detach().numpy())
index_to_word = {i: w for w, i in bow.word_to_index.items()}
predicted_word = index_to_word[output]
print("Predicted words:", predicted_word)  # Print the predicted words
from slm.bag_of_words import bag_of_words  # Import the bag_of_words class
from slm.networks import Transformer  # Import the Embedding class
import torch.optim as optim
import torch
import numpy as np

bow = bag_of_words()  # Create an instance of the bag_of_words class
line = "Alice was beginning"
vector = bow.codify(line)  # Convert the line into a bag-of-words vector

#embed = Embedding(vocab_size=len(bow.word_to_index), embedding_dim=50)  # Create an instance of the Embedding class
#out = embed(vector)  # Pass the bag-of-words vector through the embedding layer

transform = Transformer(vocab_size=len(bow.word_to_index), 
                embedding_dim=128, mlp_layers=2, mlp_dim=200,
                context_window_size=100)  # Create an instance of the Transformer class

with open('alice-in-wonderland.txt', 'r') as file:
    text = file.readlines()  # Read the text file line by line 

codified_text = [bow.codify(t) for t in text if t.strip()]  # Codify each line of the text

indices = np.arange(0, len(codified_text))
shuffled_indices = np.random.permutation(indices)  # Shuffle the indices
shuffled_codified_text = [codified_text[i] for i in shuffled_indices]  # Shuffle
test_size = int(0.5 * len(codified_text))  # Define the test size
train_text = shuffled_codified_text[:-test_size]  # Split into training text
test_text = shuffled_codified_text[-test_size:]  # Split into test text
val_size = int(0.5 * len(test_text))  # Define the validation size
test_text, val_text = test_text[:-val_size], test_text[-val_size:]

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(transform.parameters(), lr=1e-4, weight_decay=1e-5)  # Use AdamW optimizer

best_loss = float('inf')  # Initialize best loss
patience_counter = 0  # Initialize patience counter
patience = 25

for epoch in range(100):  # Number of epochs
    optimizer.zero_grad()
    total_loss = 0.0
    count = 0
    for vector in train_text:
        if len(vector) > 1:
            output = transform(vector.unsqueeze(0))  # Add batch dimension
            target = vector[1:]
            loss = criterion(output[0, :-1], target)
            total_loss += loss.item()
            count += 1
            loss.backward()
            optimizer.step()
    avg_loss = total_loss / count if count > 0 else 0

    val_loss = 0.0
    val_count = 0
    with torch.no_grad():
        for vector in val_text:
            if len(vector) > 1:
                output = transform(vector.unsqueeze(0))
                target = vector[1:]
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
    for vector in test_text:
        if len(vector) > 1:
            output = transform(vector.unsqueeze(0))
            target = vector[1:]
            loss = criterion(output[0, :-1], target)
            test_loss += loss.item()
            count += 1
    test_loss /= count if count > 0 else 0
    print(f"Test Loss: {test_loss}")  # Print the test loss

output = transform(torch.tensor(bow.codify("how are ")).unsqueeze(0))
print("Output shape:", output.shape)  # Print the shape of the output
output = np.argmax(output[0, -1, :].detach().numpy())
index_to_word = {i: w for w, i in bow.word_to_index.items()}
predicted_word = index_to_word[output]
print("Predicted words:", predicted_word)  # Print the predicted words
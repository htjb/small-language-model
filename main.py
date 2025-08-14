from slm.bag_of_words import bag_of_words  # Import the bag_of_words class
from slm.networks import Transformer  # Import the Embedding class
import torch.optim as optim
import torch
import numpy as np

bow = bag_of_words()  # Create an instance of the bag_of_words class
line = "Alice was beginning hello"
vector = bow.codify(line)  # Convert the line into a bag-of-words vector
print("Bag-of-words vector:", vector)  # Print the bag-of-words vector

print(bow.codify("hello there how are you doing today?"))  # Codify another line

print(len(vector))
print(len(bow.word_to_index))
#embed = Embedding(vocab_size=len(bow.word_to_index), embedding_dim=50)  # Create an instance of the Embedding class
#out = embed(torch.tensor(vector))  # Pass the bag-of-words vector through the embedding layer

transform = Transformer(vocab_size=len(bow.word_to_index), 
                embedding_dim=6, mlp_layers=2, mlp_dim=10,
                context_window_size=100)  # Create an instance of the Transformer class
out = transform(torch.tensor(vector))  # Pass the bag-of-words vector through the
print("Transformer output shape:", out)  # Print the shape of the output
with open('alice-in-wonderland.txt', 'r') as file:
    text = file.readlines()  # Read the text file line by line 

codified_text = [bow.codify(t) for t in text if t.strip()]  # Codify each line of the text
print("Codified text:", codified_text[:5])  # Print the first 5

indices = np.arange(0, len(codified_text))
shuffled_indices = np.random.permutation(indices)  # Shuffle the indices
shuffled_codified_text = [codified_text[i] for i in shuffled_indices]  # Shuffle
test_size = int(0.2 * len(codified_text))  # Define the test size
train_text = shuffled_codified_text[:-test_size]  # Split into training text
test_text = shuffled_codified_text[-test_size:]  # Split into test text
val_size = int(0.1 * len(train_text))  # Define the validation size
train_text, val_text = train_text[:-val_size], train_text[-val_size:]

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(transform.parameters(), lr=0.001)

for epoch in range(10):  # Number of epochs
    optimizer.zero_grad()
    total_loss = 0.0
    count = 0
    for vector in train_text:
        if len(vector) > 1:
            output = transform(torch.tensor(vector).unsqueeze(0))  # Add batch dimension
            target = torch.tensor(vector[1:])
            # one hot encode the target
            target = torch.nn.functional.one_hot(target, 
                            num_classes=len(bow.word_to_index)).float()
            loss = criterion(output[0, :-1], target)
            total_loss += loss.item()
            count += 1
            loss.backward()
            optimizer.step()
    avg_loss = total_loss / count if count > 0 else 0
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")
from slm.bag_of_words import bag_of_words  # Import the bag_of_words class
from slm.networks import Transformer  # Import the Embedding class
import torch

bow = bag_of_words()  # Create an instance of the bag_of_words class
line = "Alice was beginning"
vector = bow.codify(line)  # Convert the line into a bag-of-words vector
print("Bag-of-words vector:", vector)  # Print the bag-of-words vector

print(bow.codify("hello there how are you doing today?"))  # Codify another line

print(len(vector))
print(len(bow.word_to_index))
#embed = Embedding(vocab_size=len(bow.word_to_index), embedding_dim=50)  # Create an instance of the Embedding class
#out = embed(torch.tensor(vector))  # Pass the bag-of-words vector through the embedding layer

transform = Transformer(vocab_size=len(bow.word_to_index), 
                embedding_dim=5, mlp_layers=2, mlp_dim=10)  # Create an instance of the Transformer class
out = transform(torch.tensor(vector))  # Pass the bag-of-words vector through the


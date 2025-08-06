from slm.bag_of_words import bag_of_words  # Import the bag_of_words class

bow = bag_of_words()  # Create an instance of the bag_of_words class
line = "Alice was beginning to get very tired of sitting by her sister on the bank."
vector = bow.codify(line)  # Convert the line into a bag-of-words vector
print("Bag-of-words vector:", vector)  # Print the bag-of-words vector

print(bow.codify("hello there how are you doing today?"))  # Codify another line
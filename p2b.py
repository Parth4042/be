# Import necessary libraries
import numpy as np  # For numerical operations
from tensorflow.keras.datasets import imdb  # For loading the IMDB dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences  # For padding sequences to a fixed length
from tensorflow.keras.models import Sequential  # For defining a sequential model in Keras
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D  # For adding different layers to the model

# Load the IMDB dataset with the top 10,000 most frequent words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Print the shape of the training and test data, as well as labels
print("Train shape:", x_train.shape)  # (number of training samples, list of words in each review)
print("Test shape:", x_test.shape)    # (number of test samples, list of words in each review)
print("Train labels shape:", y_train.shape)  # (number of training labels)
print("Test labels shape:", y_test.shape)    # (number of test labels)

# Print a sample review (encoded as integers) and its label
print("\nSample encoded review:", x_train[1])  # Encoded review of the second training sample
print("Sample label (0=neg, 1=pos):", y_train[1])  # The label corresponding to the review (0 = negative, 1 = positive)

# Get the word-to-ID mapping for the dataset
vocab = imdb.get_word_index()  # Dictionary mapping word to its index (integer representation)
print("Word ID for 'the':", vocab['the'])  # Display the ID corresponding to the word 'the'

# Define class names for interpreting the output labels
class_names = ['Negative', 'Positive']

# Reverse the word-to-ID mapping to decode reviews back to text
reverse_index = {value: key for (key, value) in vocab.items()}  # Reverse the mapping (ID -> word)

def decode(review):
    """Decode an encoded review (list of integers) back to a string."""
    return " ".join([reverse_index.get(i, '?') for i in review])  # Convert indices to words, use '?' for unknown words

# Show the decoded version of the second training review
print("\nDecoded Review:\n", decode(x_train[1]))  # Decode the second review and print

# Show lengths of the reviews (number of words in each review)
def show_lengths():
    print("\nSample lengths:")
    print("Train[0]:", len(x_train[0]))  # Length of the first training review
    print("Train[1]:", len(x_train[1]))  # Length of the second training review
    print("Test[0]:", len(x_test[0]))    # Length of the first test review
    print("Test[1]:", len(x_test[1]))    # Length of the second test review

show_lengths()  # Call the function to print sample lengths

# Pad the sequences to ensure all reviews have the same length (256 words in this case)
x_train = pad_sequences(x_train, value=vocab['the'], padding='post', maxlen=256)  # Padding after each review to length 256
x_test = pad_sequences(x_test, value=vocab['the'], padding='post', maxlen=256)  # Padding for test set

# Re-show the lengths of the reviews after padding
show_lengths()  # Call the function again to print updated lengths after padding

# Define the neural network model
model = Sequential([  # Sequential model allows stacking layers linearly
    Embedding(10000, 16),  # Embedding layer: input size = 10000 (words), output size = 16 (embedding dimension)
    GlobalAveragePooling1D(),  # Pooling layer to reduce sequence dimensions, averaging over the sequence length
    Dense(16, activation='relu'),  # Fully connected layer with 16 neurons and ReLU activation
    Dense(1, activation='sigmoid')  # Output layer with 1 neuron (binary classification: 0 or 1) and sigmoid activation
])

# Compile the model specifying optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
# 'adam' optimizer is commonly used, 'binary_crossentropy' is suitable for binary classification

# Print a summary of the model architecture (layers, number of parameters, etc.)
model.summary()

# Train the model on the training data for 10 epochs with a batch size of 128
# Also use the test set for validation during training to monitor performance on unseen data
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Predict the sentiment of a specific sample from the test set (index 10)
sample_index = 10
sample_input = np.expand_dims(x_test[sample_index], axis=0)  # Expand dimensions to match model input (batch size of 1)
predicted_value = model.predict(sample_input)[0][0]  # Predict sentiment and extract the value from the output array

# Print the actual label and predicted probability for the sample
print(f"\nSample {sample_index} actual label:", y_test[sample_index])  # Print actual label (0 or 1)
print("Predicted probability:", predicted_value)  # Print predicted probability (between 0 and 1)

# Interpret the prediction
final_value = int(predicted_value > 0.5)  # Convert the predicted probability to binary label (0 or 1)
print("Predicted class:", final_value)  # Print the predicted class (0 = Negative, 1 = Positive)
print("Sentiment:", class_names[final_value])  # Map the predicted class to 'Negative' or 'Positive'

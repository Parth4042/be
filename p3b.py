# Import necessary libraries
import numpy as np  # For numerical operations, especially with arrays
import pandas as pd  # For data manipulation and loading CSV files
import matplotlib.pyplot as plt  # For plotting and visualizing data (images)
from tensorflow.keras.models import Sequential  # For defining a sequential neural network
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten  # Layers for building the CNN model

# Class labels for Fashion MNIST (10 different clothing categories)
class_name = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Load the training and testing datasets from CSV files
df1 = pd.read_csv("fashion-mnist_train.csv")  # Training data
df2 = pd.read_csv("fashion-mnist_test.csv")  # Testing data

# Split the dataset into features (x) and labels (y)
x_train = df1.drop("label", axis=1).values  # Features (pixel values) for training
y_train = df1["label"].values  # Labels (categories) for training
x_test = df2.drop("label", axis=1).values  # Features (pixel values) for testing
y_test = df2["label"].values  # Labels (categories) for testing

# Reshape the data into 28x28 images (Fashion MNIST is a dataset of 28x28 pixel images)
x_train = x_train.reshape(-1, 28, 28)  # Reshape training data to (num_samples, 28, 28)
x_test = x_test.reshape(-1, 28, 28)  # Reshape testing data to (num_samples, 28, 28)

# Optional: Show an image from the training set to visualize
plt.imshow(x_train[0], cmap='gray')  # Display the first image in grayscale
plt.title(f"Label: {class_name[y_train[0]]}")  # Display the label (class name) of the image
plt.show()  # Show the image

# Normalize the pixel values to be between 0 and 1 for better neural network performance
x_train = x_train / 255.0  # Normalize training data
x_test = x_test / 255.0  # Normalize testing data

# Reshape the data to include the channel dimension (for grayscale, it's 1 channel)
x_train = x_train.reshape(-1, 28, 28, 1)  # Reshape to (num_samples, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)  # Reshape to (num_samples, 28, 28, 1)

# Build the Convolutional Neural Network (CNN) model
model = Sequential([  # Initialize a Sequential model
    Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 1st convolutional layer with 64 filters, 3x3 kernel
    MaxPooling2D((2, 2)),  # Max pooling layer to down-sample the image
    Conv2D(64, (3, 3), activation='relu'),  # 2nd convolutional layer with 64 filters, 3x3 kernel
    MaxPooling2D((2, 2)),  # Max pooling layer
    Flatten(),  # Flatten the 2D feature map into a 1D vector
    Dense(128, activation='relu'),  # Fully connected layer with 128 neurons
    Dense(10, activation='softmax')  # Output layer with 10 neurons for 10 classes, softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam',  # Optimizer used for training (Adam)
              loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification
              metrics=['accuracy'])  # Metric to track during training

# Print the model summary (architecture)
model.summary()

# Train the model on the training data and validate it on the test data
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model on the test dataset to get the loss and accuracy
loss, accuracy = model.evaluate(x_test, y_test)
print("Loss:", loss)  # Print the loss value
print("Accuracy:", accuracy * 100)  # Print the accuracy as a percentage

# Predict on the test data
index = 10  # Choose an index of a test image to predict
predictions = model.predict(x_test)  # Get the predicted probabilities for all test images

# Get the predicted class by taking the index of the highest probability
final_value = np.argmax(predictions[index])  # The predicted class with the highest probability

# Print the predicted probabilities, actual label, and predicted label
print("Predicted probabilities:", predictions[index])  # Print the predicted class probabilities
print("Actual label:", y_test[index])  # Print the actual label of the image
print("Predicted label:", final_value)  # Print the predicted class index
print("Class Label:", class_name[final_value])  # Print the class name corresponding to the predicted index

# Visualize the test image and show the predicted label
plt.imshow(x_test[index].reshape(28, 28), cmap='gray')  # Display the test image
plt.title(f"Predicted: {class_name[final_value]}")  # Display the predicted class name as the title
plt.show()  # Show the image

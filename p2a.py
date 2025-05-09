# Import required libraries
import pandas as pd  # For data loading and manipulation
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting the dataset into train/test sets
from sklearn.preprocessing import LabelEncoder  # For encoding categorical labels into numerical values
from tensorflow.keras.models import Sequential  # For defining a Sequential neural network model
from tensorflow.keras.layers import Dense, Dropout  # For adding fully connected and dropout layers to the model

# Define column names based on dataset description
columns = [
    "lettr", "x-box", "y-box", "width", "height", "onpix", "x-bar", "y-bar", "x2bar", "y2bar",
    "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"
]

# Load the letter recognition dataset into a pandas DataFrame
df = pd.read_csv('letter-recognition.data', names=columns)  # Dataset contains information about letters and their features
print(df.head())  # Display the first 5 rows of the dataset for inspection

# Split the data into features (x) and target labels (y)
x = df.drop("lettr", axis=1).values  # Features: all columns except 'lettr' (the label)
y = df["lettr"].values  # Target: the 'lettr' column containing the letter labels

# Display the shapes of features and labels, and the unique classes in the target variable
print("Feature shape:", x.shape)  # Number of samples and features
print("Label shape:", y.shape)  # Number of samples and target labels
print("Unique classes:", np.unique(y))  # Display all unique letters in the dataset

# Split the dataset into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Normalize the features (scale the pixel values to be between 0 and 1)
x_train = x_train / 255.0  # Normalize training data
x_test = x_test / 255.0  # Normalize testing data

# Encode class labels (letters) into numeric values (0-25 for A-Z)
encoder = LabelEncoder()  # Initialize label encoder
y_train = encoder.fit_transform(y_train)  # Encode training labels
y_test = encoder.transform(y_test)  # Encode testing labels

# Define the list of class names (A-Z) for final label mapping
class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # Create a list of characters A-Z

# Build a neural network model
model = Sequential([  # Sequential model where layers are stacked on top of each other
    Dense(512, activation='relu', input_shape=(16,)),  # First dense layer with 512 neurons and ReLU activation
    Dropout(0.2),  # Dropout layer to prevent overfitting (20% chance of dropping each neuron during training)
    Dense(256, activation='relu'),  # Second dense layer with 256 neurons and ReLU activation
    Dropout(0.2),  # Another dropout layer with 20% probability
    Dense(26, activation='softmax')  # Output layer with 26 neurons (one for each letter A-Z) and softmax activation
])

# Compile the model with optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  
# - optimizer='adam': Adaptive optimizer that works well with neural networks
# - loss='sparse_categorical_crossentropy': Suitable for multi-class classification problems with integer labels
# - metrics=['accuracy']: Track accuracy as the evaluation metric during training

# Display the model summary: layers, number of parameters, etc.
model.summary()

# Train the model with the training data for 50 epochs and a batch size of 128
# Use validation data to monitor performance on unseen test data during training
model.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test))

# Make predictions on the test data
predictions = model.predict(x_test)

# Display prediction for a specific index (e.g., index 10) in the test set
index = 10
pred = model.predict(x_test[index].reshape(1, -1))  # Reshape the input to match model input format (batch size of 1)
final_value = np.argmax(pred)  # Extract the predicted class by finding the index of the highest probability

# Print the actual vs. predicted value for that sample
print(f"\nSample {index} Prediction:")
print("Actual label:", y_test[index])  # Actual label (letter)
print("Predicted label:", final_value)  # Predicted numeric label (0-25)
print("Predicted Class (A-Z):", class_names[final_value])  # Convert the predicted numeric label back to a letter

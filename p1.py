# Import required libraries
import pandas as pd  # For data loading and manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualizations (not used in this script but useful for exploration)

# Load and explore the dataset
BostonTrain = pd.read_csv("boston_test.csv")  # Load the dataset from a CSV file into a pandas DataFrame
print(BostonTrain.head())  # Display the first 5 rows of the dataset to understand its structure
print(BostonTrain.info())  # Print column-wise info: types, non-null counts, etc.
print(BostonTrain.describe())  # Show statistical summary (mean, std, min, max, etc.) of each feature

# Prepare input (features) and output (target) data
X = BostonTrain.iloc[:, 1:-1].values  # Select all columns except the first (often an ID) and last (target)
Y = BostonTrain.iloc[:, -1].values    # Select the last column as the target variable (e.g., housing price)

# Split the dataset into training and test sets (60% train, 40% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)

# Import necessary components for building a neural network
from tensorflow.keras.models import Sequential  # Model type where layers are stacked sequentially
from tensorflow.keras.layers import Dense  # Fully connected (dense) neural network layer

# Define a sequential model with 3 layers
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # First hidden layer with 128 neurons, ReLU activation
    Dense(64, activation='relu'),  # Second hidden layer with 64 neurons, ReLU activation
    Dense(1, activation='linear')  # Output layer with 1 neuron (regression), linear activation
])

# Compile the model with optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  
# - optimizer='adam': efficient gradient-based optimizer
# - loss='mse': mean squared error for regression
# - metrics=['mae']: mean absolute error as additional metric

# Display the model architecture and number of parameters
model.summary()

# Train the model on training data for 100 epochs with batch size 1
# Also validate on test data after each epoch
model.fit(X_train, y_train, epochs=100, batch_size=1, validation_data=(X_test, y_test))

# Make a prediction for a single test sample (e.g., index 8)
sample_index = 8
sample = np.array([X_test[sample_index]])  # Reshape the single sample for prediction
actual = y_test[sample_index]  # Actual value for the selected sample
predicted = model.predict(sample)[0][0]  # Predict using the model and extract scalar value

# Print the actual vs. predicted value for that sample
print(f"\nSample {sample_index}:")
print("Actual Value: ", actual)
print("Predicted Value: ", predicted)

# Import evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predict housing prices for the entire test set
y_pred = model.predict(X_test)

# Compute evaluation metrics to assess model performance
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
r2 = r2_score(y_test, y_pred)  # R² Score: proportion of variance explained by the model

# Print out evaluation metrics
print("\nEvaluation Metrics:")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R² Score:", r2)

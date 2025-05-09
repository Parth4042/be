# Import necessary libraries
import pandas as pd  # For handling data (loading CSV files, data manipulation)
import numpy as np  # For numerical operations (like reshaping arrays)
from sklearn.preprocessing import MinMaxScaler  # For normalizing the data
from tensorflow.keras.models import Sequential  # For creating the neural network
from tensorflow.keras.layers import LSTM, Dense  # For adding LSTM and Dense layers

# Load training and testing data
df1 = pd.read_csv("Google_stock_price_train.csv")  # Load the training data CSV
df2 = pd.read_csv("Google_stock_price_test.csv")  # Load the testing data CSV

# Display the information about the training data (to check the structure of the data)
print("Train data info:")
print(df1.info())  # Print the summary of the training dataset (column names, types, non-null values)

# Clean the 'Close' column to remove commas and convert it to a float
df1['Close'] = df1['Close'].astype(str).str.replace(",", "").astype(float)  # Remove commas and convert to float
df2['Close'] = df2['Close'].astype(str).str.replace(",", "").astype(float)  # Apply the same cleaning to test data

# Normalize the 'Close' column to scale values between 0 and 1
train_scaler = MinMaxScaler()  # Initialize the MinMaxScaler for training data
df1['Normalized close'] = train_scaler.fit_transform(df1['Close'].values.reshape(-1, 1))  # Fit and transform training data

test_scaler = MinMaxScaler()  # Initialize the MinMaxScaler for test data
df2['Normalized close'] = test_scaler.fit_transform(df2['Close'].values.reshape(-1, 1))  # Fit and transform test data

# Prepare the sequences for training and testing (LSTM requires 3D input)
# x_train and y_train are sequences for the model to learn the pattern
x_train = df1['Normalized close'].values[:-1].reshape(-1, 1, 1)  # Input sequence for training (all values except the last)
y_train = df1['Normalized close'].values[1:].reshape(-1, 1, 1)  # Target sequence for training (all values except the first)

x_test = df2['Normalized close'].values[:-1].reshape(-1, 1, 1)  # Input sequence for testing (all values except the last)
y_test = df2['Normalized close'].values[1:].reshape(-1, 1, 1)  # Target sequence for testing (all values except the first)

# Define the LSTM model
model = Sequential()  # Initialize a Sequential model

# Add an LSTM layer with 4 units, input shape of (1, 1) since each sample has one feature (the price at each time step)
model.add(LSTM(4, input_shape=(1, 1)))  

# Add a Dense layer to output the predicted price (1 unit)
model.add(Dense(1))  # Dense layer with 1 unit as we are predicting a single value

# Compile the model with Adam optimizer and Mean Squared Error (MSE) loss
model.compile(optimizer='adam', loss='mse')  # 'adam' optimizer and 'mse' loss function (used for regression tasks)

# Print the model summary to view the architecture
model.summary()

# Train the model for 100 epochs with a batch size of 1, using training and validation data
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=1)

# Evaluate the model on the test data to get the loss value
test_loss = model.evaluate(x_test, y_test)  # Evaluate on the test data
print("Testing Loss:", test_loss)  # Print the test loss

# Predict on the test data using the trained model
pred = model.predict(x_test)  # Get predictions for the test data

# Reverse the scaling to get actual price values (from normalized values back to original values)
y_test_actual = test_scaler.inverse_transform(y_test.reshape(-1, 1))  # Inverse transform the actual test values
y_test_pred = test_scaler.inverse_transform(pred.reshape(-1, 1))  # Inverse transform the predicted values

# Display a sample prediction and its corresponding actual value
index = 1  # Choose an index for a sample prediction
print(f"Actual value at index {index}:", y_test_actual[index])  # Print the actual value at index 1
print(f"Predicted value at index {index}:", y_test_pred[index])  # Print the predicted value at index 1

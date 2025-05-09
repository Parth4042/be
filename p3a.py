# Import necessary libraries
import numpy as np  # For numerical operations
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array  # For loading and processing images
from tensorflow.keras.models import Sequential  # For defining a sequential model in Keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization  # CNN layers

# Define the directories where the training and validation images are stored
train_dir = "./archive/tomato/train"  # Path to the training images
val_dir = "./archive/tomato/val"  # Path to the validation images

# Set constants for image size, batch size, and number of epochs
img_size = 224  # Resize all images to 224x224 pixels
batch_size = 32  # Number of images processed in each batch during training
epochs = 2  # Number of epochs for training (you can increase this for better performance)

# Data preprocessing: normalize the images to scale pixel values between 0 and 1
train_datagen = ImageDataGenerator(rescale=1./255)  # For training data
val_datagen = ImageDataGenerator(rescale=1./255)  # For validation data

# Prepare the training data by loading images from the directory and resizing them to the target size
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(img_size, img_size),  # Resize images to 224x224
    batch_size=batch_size,  # Number of images in each batch
    class_mode='categorical'  # Multi-class classification (multiple diseases)
)

# Prepare the validation data in the same way
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(img_size, img_size),  # Resize images to 224x224
    batch_size=batch_size,  # Number of images in each batch
    class_mode='categorical'  # Multi-class classification
)

# Define the CNN model architecture
model = Sequential([  # Sequential model for stacking layers in order
    # First convolutional layer with 32 filters, 3x3 kernel, ReLU activation function, and input size of (224, 224, 3) (RGB images)
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    BatchNormalization(),  # Batch normalization for stabilizing and speeding up training
    MaxPooling2D(2, 2),  # Max pooling layer to reduce spatial dimensions (2x2 pool size)

    # Second convolutional layer with 64 filters, 3x3 kernel, ReLU activation function
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Third convolutional layer with 64 filters, 3x3 kernel, ReLU activation function
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Fourth convolutional layer with 128 filters, 3x3 kernel, ReLU activation function
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Flatten the 3D output of the last pooling layer into a 1D vector
    Flatten(),
    Dense(128, activation='relu'),  # Fully connected layer with 128 neurons and ReLU activation
    Dropout(0.2),  # Dropout to prevent overfitting (20% chance of dropping a neuron)
    Dense(64, activation='relu'),  # Fully connected layer with 64 neurons and ReLU activation
    Dense(train_generator.num_classes, activation='softmax')  # Output layer with a number of neurons equal to the number of classes (diseases), softmax for multi-class classification
])

# Compile the model with the Adam optimizer, categorical cross-entropy loss, and accuracy metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model architecture (layers, number of parameters, etc.)
model.summary()

# Train the model using the training data and validate using the validation data for the specified number of epochs
model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Evaluate the model's performance on the validation set and print the loss and accuracy
loss, accuracy = model.evaluate(val_generator)
print("Loss:", loss)
print("Accuracy:", accuracy * 100)  # Multiply by 100 to display accuracy as a percentage

# Test prediction with an example image
img_path = "./archive/tomato/val/Tomato___Septoria_leaf_spot/0a25f893-1b5f-4845-baa1-f68ac03d96ac___Matt.S_CG 7863.jpg"
img = load_img(img_path, target_size=(img_size, img_size))  # Load the image and resize it to 224x224
img_array = img_to_array(img)  # Convert the image to a NumPy array
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Add batch dimension and normalize the pixel values

# Predict the class of the image
prediction = model.predict(img_array)  # Make a prediction for the input image
predicted_class = np.argmax(prediction)  # Get the index of the highest probability (the predicted class)

# Class names corresponding to the labels
class_name = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Print the prediction vector, predicted class index, and class name
print("Prediction vector:", prediction)  # Display the predicted probabilities for each class
print("Predicted class index:", predicted_class)  # Display the index of the predicted class
print("Class Name:", class_name[predicted_class])  # Display the predicted class name

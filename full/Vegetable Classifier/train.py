from PyQt5 import QtCore, QtGui, QtWidgets
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


data = []
labels = []
cur_path = os.getcwd()  # To get current directory

# Mapping class labels
classs = {
    1: "Bean",
    2: "Bitter_Gourd",
    3: "Bottle_Gourd",
    4: "Brinjal",
    5: "Broccoli",
    6: "Cabbage",
    7: "Capsicum",
    8: "Carrot",
    9: "Cauliflower",
    10: "Cucumber",
    11: "Papaya",
    12: "Potato",
    13: "Pumpkin",
    14: "Radish",
    15: "Tomato"
}

classes = len(classs)

# Retrieving the images and their labels
print("Obtaining Images & their Labels...")
for i, veg_name in classs.items():
    path = os.path.join(cur_path, 'Vegetable Images', 'train', veg_name)
    path = path.replace('\\', '/')  # Normalize path
    
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        continue

    images = os.listdir(path)

    for a in images:
        try:
            image_path = os.path.join(path, a)
            image = Image.open(image_path)
            image = image.resize((30, 30))
            image = np.array(image)
            if image.shape == (30, 30, 3):
                data.append(image)
                labels.append(i - 1)
                print(f"{a} Loaded")
        except Exception as e:
            print(f"Error loading image {a}: {e}")

print("Dataset Loaded")

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

if data.size == 0 or labels.size == 0:
    raise ValueError("No images loaded. Please check your folder paths and data.")

# Normalize data
data = data / 255.0

print(data.shape, labels.shape)

# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Converting the labels into one hot encoding
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

# Model building
print("Training under process...")
model = Sequential([
    Conv2D(32, (5, 5), activation='relu', input_shape=X_train.shape[1:]),
    Conv2D(32, (5, 5), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(classes, activation='softmax')
])

print("Initialized model")

# Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))


# Save model and visualization
model.save("my_model.h5")


print("Saved Model")

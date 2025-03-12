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
classes = 4
cur_path = os.getcwd() #To get current directory


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

print(data.shape, labels.shape)
#Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Converting the labels into one hot encoding
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

def classify(img_file):
    model = load_model('my_model.h5')
    print("Loaded model from disk");
    path2=img_file
    print(path2)
    test_image = Image.open(path2)
    test_image = test_image.resize((30, 30))
    test_image = np.expand_dims(test_image, axis=0)
    test_image = np.array(test_image)
    #result = model.predict_classes(test_image)[0]	
    predict_x=model.predict(test_image)
    result=np.argmax(predict_x,axis=1)
    sign = classs[int(result) + 1]        
    print(sign)


import os
path = 'Dataset\\test\\'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
   for file in f:
     if '.jpg' in file:
       files.append(os.path.join(r, file))

for f in files:
   classify(f)
   print('\n')
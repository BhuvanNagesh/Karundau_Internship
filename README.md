# 🌟 My Project Portfolio

Welcome to my project portfolio repository! This is a complete web-based application built using **HTML**, **CSS**, and **Django**. The application consists of a navigation bar with four main sections, each showcasing different aspects of my work, including my internship experience, machine learning models, and image recognition projects. 

---

## 🛠️ **Tech Stack:**
- **Frontend:** HTML, CSS, Bootstrap
- **Backend:** Django (Python)
- **Database:** SQLite (or your preferred database)
- **Libraries & Tools:** Matplotlib, Seaborn, NumPy, Pandas

---

## 🔗 **Project Contents:**

### 1. 🏢 **About My Internship Company**
This section provides insights into my internship at **Karunadu Technologies**, including the company's mission, my learning journey, and the skills I gained during the experience.

### 2. 🚀 **Astral Classification using Spectral Characteristics using Naive Bayes Algorithm**
An ML project that classifies stellar objects (stars, galaxies, or quasars) using space observation data from the **Sloan Digital Sky Survey (SDSS)**.

- **Key Features:**
  - Right Ascension & Declination angles
  - Ultraviolet, Green, Red, Near Infrared, and Infrared filters
  - Redshift value, Plate ID, MJD, and Fiber ID
  - Class label (Star, Galaxy, or Quasar)

### 3. 📊 **Using Naive Bayes Algorithm**
Implemented the **Naive Bayes algorithm** for classification tasks. Despite its simplicity, this algorithm is effective for large datasets with categorical features.

- **Applications:**
  - Spam detection
  - Sentiment analysis
  - Document classification

### 4. 🥦 **Vegetable Classifier**
A machine learning project to classify vegetables into 15 distinct categories using image data i download from kaggle.

- **Vegetables Classified:** Bean, Bitter Gourd, Brinjal, Broccoli, Tomato, and more!
- **Use Cases:** Smart farming, automated grocery systems
## 📊 Algorithm: Convolutional Neural Network (CNN)

The core of the image classification system is a CNN, a deep learning model tailored for image recognition tasks. Let’s break it down! ⚡

```python
# CNN Architecture
model = Sequential([
    Conv2D(32, (5, 5), activation='relu', input_shape=X_train.shape[1:]),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(classes, activation='softmax')
])
```

-**🔍 Layer-by-Layer Breakdown

1. **Convolutional Layer (Conv2D):**
   - Applies 32 filters of size `(5x5)` to the input image.
   - Detects features like edges, textures, and shapes.

2. **Activation Function:**
   - `ReLU` (Rectified Linear Unit) replaces negative values with zero.
   - Speeds up convergence and adds non-linearity.

3. **Pooling Layer (MaxPooling):**
   - `MaxPool2D(pool_size=(2, 2))` downsamples the image.
   - Reduces dimensionality and computation by selecting the maximum value in each block.

4. **Dropout:**
   - `Dropout(0.25)` randomly drops 25% of neurons during training.
   - Prevents overfitting and enhances generalization.

5. **Flatten Layer:**
   - Converts the 3D tensor into a 1D vector.
   - Prepares data for the fully connected layers.

6. **Fully Connected (Dense) Layers:**
   - `Dense(256, activation='relu')`: A layer with 256 neurons.
   - `Dense(classes, activation='softmax')`: The output layer with neurons equal to the number of classes, using `softmax` to output class probabilities.

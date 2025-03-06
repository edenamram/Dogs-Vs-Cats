# Dogs vs. Cats Classification Project

This project uses the Kaggle Dogs vs. Cats dataset to build an image classification model. The project covers end-to-end data processing, exploratory data analysis (EDA), model building, dimensionality reduction using PCA, evaluation, and fine-tuning.

## Table of Contents
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Building](#model-building)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Model Evaluation](#model-evaluation)
- [Fine-Tuning and Hyperparameter Tuning](#fine-tuning-and-hyperparameter-tuning)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Dataset
- **Source:** Kaggle Dogs vs. Cats dataset.
- **Description:** Contains images of dogs and cats with filenames indicating the class (e.g., `cat.1234.jpg` or `dog.5678.jpg`).

## Data Preprocessing
1. **Resizing and Normalization:**
   - Images are resized to `64x64` and normalized so that pixel values are in the range `[0,1]`.
   - Example function:
     ```python
     def preprocess_image(img_path):
         img = load_img(img_path, target_size=(64, 64))  # Resize image to 64x64
         img = img_to_array(img) / 255.0                  # Normalize pixels
         return img.astype(np.float16)                    # Convert data type for memory efficiency
     ```
2. **Data Splitting:**
   - The processed images (stored in `processed_images`) and their labels (derived from filenames) are split into training, validation, and test sets.
3. **Data Cleaning & EDA:**
   - Corrupted or low-quality images are detected and removed.
   - Basic EDA (e.g., class distribution, pixel intensity statistics) is performed.

## Exploratory Data Analysis (EDA)
- **Visual Inspection:** Randomly selected images are displayed to verify correctness.
- **Class Distribution:** Bar plots (using Seaborn) show the number of images per class.
- **Pixel Statistics:** Mean and standard deviation of pixel intensities are computed to validate normalization.

## Model Building
### Convolutional Neural Network (CNN)
The CNN model is built using TensorFlow/Keras with the following architecture:
- **Convolutional Layers:** Three Conv2D layers with increasing numbers of filters (32, 64, 128) and ReLU activations.
- **Pooling Layers:** MaxPooling2D layers reduce spatial dimensions.
- **Flatten:** Converts the feature maps into a 1D vector.
- **Dense Layers:** A fully connected layer with 512 units, followed by a Dropout layer (0.5) and a final Dense layer with sigmoid activation for binary classification.
  
Example:
```python
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
```
## Dimensionality Reduction
### PCA:
The preprocessed images are flattened and then transformed using PCA to reduce the number of features.
Example:
     ```python
            num_samples = processed_images.shape[0]
            images_flat = processed_images.reshape(num_samples, -1)  # Flatten images
            from sklearn.decomposition import PCA
            pca = PCA(n_components=300, random_state=42)
            images_pca = pca.fit_transform(images_flat)
     ```
The PCA-reduced data is then split and used to train classical models (e.g., a Decision Tree).






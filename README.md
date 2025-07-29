
### Upload Kaggle API Key (`kaggle.json`)

1. Go to your [Kaggle Account Settings](https://www.kaggle.com/account)
2. Create a new API Token
3. Upload the `kaggle.json` file when prompted

---

## ⚙️ Project Workflow

### 1. Environment Setup and Data Download
- Sets up Kaggle credentials
- Downloads and extracts the dataset

### 2. Data Loading and Preprocessing
- Loads images with `image_dataset_from_directory`
- Preprocesses using MobileNetV2 normalization
- Uses caching and prefetching for performance
- Splits data into train (70%), validation (15%), and test (15%)

### 3. Data Visualization
- Displays a grid of 16 random training images

### 4. Model Architecture
- MobileNetV2 base model (frozen)
- Global Average Pooling
- Dense layers with ReLU and Dropout
- Output layer with softmax activation

### 5. Training
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Callbacks:
  - Early stopping
  - Model checkpoint (save best weights)

### 6. Evaluation
- Final test accuracy
- Accuracy and loss curves
- Classification report
- Confusion matrix

### 7. Saving the Model
- Final model saved as `.keras` format
- Downloaded automatically in Colab

---

## 🧠 Model Summary

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Head**:
  - `GlobalAveragePooling2D`
  - `Dense(128, activation='relu')`
  - `Dropout(0.4)`
  - `Dense(2, activation='softmax')`

---

## 🧪 Evaluation Metrics

- Accuracy
- Loss
- Precision, Recall, F1-score
- Confusion Matrix

---

## 📊 Example Outputs

The notebook includes:
- Plots of training vs validation accuracy/loss
- Confusion matrix heatmap
- Detailed classification metrics per class

---

## 💾 Output Files

- `fire_model.weights.h5`: Best model weights
- `fire_detection_model_v2.keras`: Final full model (for deployment)

---

## 🚀 Future Work

- Fine-tune unfrozen layers of MobileNetV2
- Add data augmentation (e.g. rotation, zoom)
- Experiment with other architectures (ResNet, EfficientNet)
- Deploy on mobile or edge devices (e.g. TFLite)

---

## 📚 References

- [Kaggle: Fire Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow Documentation](https://www.tensorflow.org/)

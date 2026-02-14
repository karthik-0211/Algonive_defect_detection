# NEU Surface Defect Detection using CNN

A deep learning project for automated surface defect detection on steel surfaces using Convolutional Neural Networks (CNN) with the NEU Surface Defect Database.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [Results](#results)
- [Logging and Monitoring](#logging-and-monitoring)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a Convolutional Neural Network (CNN) for automated defect detection in steel surfaces. The model is trained to classify six different types of surface defects commonly found in manufacturing processes. The solution provides real-time defect classification with confidence scores and comprehensive logging capabilities.

## 📊 Dataset

### NEU Surface Defect Database
**Source:** [Kaggle - NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)

The dataset contains grayscale images of six kinds of typical surface defects of hot-rolled steel strip:

1. **Crazing (Cr)** - Fine cracks on the surface
2. **Inclusion (In)** - Non-metallic inclusions
3. **Patches (Pa)** - Irregular surface patches
4. **Pitted Surface (PS)** - Small pits or holes
5. **Rolled-in Scale (RS)** - Scale rolled into the surface
6. **Scratches (Sc)** - Linear surface marks

### Dataset Specifications
- **Total Images:** 1,800 grayscale images
- **Images per Class:** 300 images
- **Image Dimensions:** 200×200 pixels
- **Format:** .jpg
- **Color Space:** Grayscale (converted to RGB for training)

## 🏗️ Model Architecture

### CNN Architecture Details

```
Input Layer: (224, 224, 3) RGB images

Conv Block 1:
- Conv2D: 32 filters, 3×3 kernel, ReLU activation
- BatchNormalization
- MaxPooling2D: 2×2
- Dropout: 0.25

Conv Block 2:
- Conv2D: 64 filters, 3×3 kernel, ReLU activation
- BatchNormalization
- MaxPooling2D: 2×2
- Dropout: 0.25

Conv Block 3:
- Conv2D: 128 filters, 3×3 kernel, ReLU activation
- BatchNormalization
- MaxPooling2D: 2×2
- Dropout: 0.25

Dense Layers:
- Flatten
- Dense: 256 units, ReLU activation
- Dropout: 0.5
- Dense: 128 units, ReLU activation
- Dropout: 0.5
- Output Dense: 6 units, Softmax activation
```

### Key Features of the Architecture
- **Batch Normalization:** Improves training stability and speed
- **Dropout Layers:** Prevents overfitting (dropout rates: 0.25 in conv blocks, 0.5 in dense layers)
- **Progressive Feature Learning:** Gradually increases filter depth (32 → 64 → 128)
- **Data Augmentation:** Real-time augmentation during training

## ✨ Features

- **Multi-Class Classification:** Detects 6 different types of surface defects
- **Real-Time Prediction:** Fast inference for production environments
- **Data Augmentation:** Improves model generalization
  - Rotation: ±20 degrees
  - Width/Height Shift: ±20%
  - Shear and Zoom transformations
  - Horizontal and Vertical flipping
- **Comprehensive Evaluation:** Multiple metrics for performance assessment
- **Prediction Logging:** Automated logging system for tracking predictions
- **Visualization Tools:** Includes confusion matrix and classification reports
- **Model Persistence:** Save and load trained models

## 🔧 Requirements

### Dependencies

```python
tensorflow==2.19.0
opencv-python==4.13.0.92
matplotlib==3.10.0
seaborn==0.13.2
scikit-learn==1.6.1
pandas==2.2.2
numpy>=1.26.0
pillow
```

### Hardware Requirements
- **Recommended:** GPU (CUDA-compatible) for training
- **Minimum RAM:** 8GB
- **Storage:** ~500MB for dataset + model files

## 📥 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/neu-defect-detection.git
cd neu-defect-detection
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages
```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-learn pandas
```

### 4. Download Dataset
Download the NEU Surface Defect Database from [Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)

### 5. Setup Dataset Structure
```
NEU-DET/
├── IMAGES/
│   ├── crazing/
│   ├── inclusion/
│   ├── patches/
│   ├── pitted_surface/
│   ├── rolled-in_scale/
│   └── scratches/
└── ANNOTATIONS/
```

## 📁 Project Structure

```
neu-defect-detection/
├── Defect_Detection.ipynb    # Main notebook with complete pipeline
├── README.md                  # Project documentation
├── models/                    # Saved model files
│   └── defect_model.h5
├── logs/                      # Prediction logs
│   └── prediction_logs.csv
├── data/                      # Dataset directory
│   └── NEU-DET/
├── results/                   # Training results and plots
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── classification_report.txt
└── requirements.txt           # Python dependencies
```

## 🚀 Usage

### Training the Model

```python
# 1. Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 2. Set parameters
img_size = 224
batch_size = 32
epochs = 50

# 3. Setup data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

# 4. Load data
train_generator = train_datagen.flow_from_directory(
    'data/NEU-DET/IMAGES',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# 5. Train model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)
```

### Making Predictions

```python
import cv2
import numpy as np

# Load trained model
model = keras.models.load_model('models/defect_model.h5')

# Load and preprocess image
img = cv2.imread('test_image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224))
img_normalized = img_resized / 255.0
img_batch = np.expand_dims(img_normalized, axis=0)

# Predict
prediction = model.predict(img_batch)
class_idx = np.argmax(prediction)
confidence = np.max(prediction)

class_names = ['crazing', 'inclusion', 'patches', 
               'pitted_surface', 'rolled-in_scale', 'scratches']
print(f"Prediction: {class_names[class_idx]}")
print(f"Confidence: {confidence:.4f}")
```

### Batch Prediction
```python
# Upload and predict multiple images
from google.colab import files

uploaded = files.upload()

for file_name in uploaded.keys():
    # Load and preprocess
    img = cv2.imread(file_name)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Predict
    pred = model.predict(img_batch)
    cls = class_names[np.argmax(pred)]
    conf = np.max(pred)
    
    print(f"{file_name}: {cls} ({conf:.4f})")
    
    # Visualize
    cv2.putText(img, f"{cls} ({conf:.2f})", 
                (20,50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,255,0), 2)
    cv2_imshow(img)
```

## 🎓 Training Details

### Hyperparameters
- **Image Size:** 224×224 pixels
- **Batch Size:** 32
- **Epochs:** 50 (with early stopping)
- **Learning Rate:** 0.001 (with reduction on plateau)
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy

### Callbacks
1. **Early Stopping**
   - Monitors: Validation loss
   - Patience: 10 epochs
   - Restores best weights

2. **Learning Rate Reduction**
   - Monitors: Validation loss
   - Factor: 0.5
   - Patience: 5 epochs
   - Minimum LR: 1e-7

3. **Model Checkpoint**
   - Saves best model based on validation accuracy
   - File: `defect_model_best.h5`

### Data Split
- **Training Set:** 80% (1,440 images)
- **Validation Set:** 20% (360 images)

## 📈 Evaluation

### Metrics
- **Accuracy:** Overall classification accuracy
- **Precision:** Per-class precision scores
- **Recall:** Per-class recall scores
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Detailed classification breakdown

### Evaluation Code
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Generate predictions
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes

# Classification Report
print(classification_report(y_true, y_pred_classes, 
                          target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

## 📊 Results

### Model Performance
*(Add your actual results after training)*

```
Expected Performance:
- Training Accuracy: ~95-98%
- Validation Accuracy: ~90-95%
- Test Accuracy: ~88-93%
```

### Class-wise Performance
- **Crazing:** High accuracy (typically 92-96%)
- **Inclusion:** Moderate accuracy (85-90%)
- **Patches:** High accuracy (90-95%)
- **Pitted Surface:** Moderate accuracy (85-92%)
- **Rolled-in Scale:** High accuracy (90-95%)
- **Scratches:** High accuracy (92-97%)

### Visualization Examples
- Training/Validation Loss curves
- Training/Validation Accuracy curves
- Confusion Matrix heatmap
- Sample predictions with confidence scores

## 📝 Logging and Monitoring

### Prediction Logging
The system automatically logs all predictions to `prediction_logs.csv`:

```python
def log_prediction(image_name, prediction, confidence):
    """Log prediction details to CSV file"""
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_name': image_name,
        'prediction': prediction,
        'confidence': float(confidence)
    }
    
    log_df = pd.DataFrame([log_entry])
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    log_file = 'logs/prediction_logs.csv'
    
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, mode='w', header=True, index=False)
```

### Log Format
```csv
timestamp,image_name,prediction,confidence
2024-02-14 10:30:45,image1.jpg,scratches,0.9823
2024-02-14 10:31:12,image2.jpg,crazing,0.8976
```

## 🔮 Future Improvements

### Short-term
- [ ] Implement test-time augmentation (TTA)
- [ ] Add ensemble model predictions
- [ ] Create web-based inference API (Flask/FastAPI)
- [ ] Implement explainability with Grad-CAM

### Medium-term
- [ ] Transfer learning with pre-trained models (ResNet, EfficientNet)
- [ ] Multi-scale feature extraction
- [ ] Real-time video stream defect detection
- [ ] Mobile deployment (TensorFlow Lite)

### Long-term
- [ ] Few-shot learning for rare defect types
- [ ] Anomaly detection for unknown defect patterns
- [ ] Integration with manufacturing execution systems (MES)
- [ ] Edge deployment for on-device inference

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide for Python code
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset:** NEU Surface Defect Database by Northeastern University (NEU)
- **Kaggle Dataset:** [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
- **Reference Paper:** 
  - Song, K., & Yan, Y. (2013). A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects. Applied Surface Science, 285, 858-864.

## 📧 Contact

For questions, issues, or suggestions, please:
- Open an issue on GitHub
- Contact: gudakarthikreddy2005@gmail.com

## 📚 References

1. NEU Surface Defect Database: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
2. TensorFlow Documentation: https://www.tensorflow.org/
3. Deep Learning for Computer Vision: https://www.deeplearningbook.org/

---

**Note:** This is an educational/research project. For production deployment, additional considerations for robustness, scalability, and safety are required.

**Last Updated:** February 2024

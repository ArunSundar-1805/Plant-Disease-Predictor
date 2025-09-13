# CropDoc Plant Disease Predictor

A deep learning-based application designed to automatically detect plant diseases from leaf images. This project leverages convolutional neural networks (CNNs) to classify diseases in various crop plants, aiming to assist farmers and researchers in early identification and management of plant health.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Usage](#usage)
- [Requirements](#requirements)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

Plant diseases can cause significant reduction in yield and quality. This repository provides a machine learning approach for classifying plant diseases from images using a trained CNN model. The system is built with Python, Keras, and TensorFlow, and includes Jupyter Notebooks for training and testing, as well as a Python script for running predictions.

---

## Features

- Image-based disease detection for multiple crop plants
- Pre-trained CNN model included (`trained.h5` and `trained.keras`)
- Jupyter notebooks for training (`Train_plant_disease.ipynb`) and testing (`Test_Disease.ipynb`)
- Easy-to-use prediction script (`main.py`)
- Customizable for additional crops/diseases
- Modular code for rapid experimentation

---

## Project Structure

```
├── Plant_Disease_Dataset/    # Dataset folder (images for training/testing)
├── Train_plant_disease.ipynb # Notebook for training the model
├── Test_Disease.ipynb        # Notebook for testing the model
├── main.py                   # Script for running predictions
├── trained.h5                # Pre-trained model (Keras HDF5 format)
├── trained.keras             # Pre-trained model (Keras format)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
```

---

## Dataset

- The dataset is located in the `Plant_Disease_Dataset` directory.
- Organize images into subfolders by disease type (e.g. `Plant_Disease_Dataset/Tomato___Early_blight/`).
- Each subfolder should contain images representing a specific disease or healthy leaves.
- Example dataset sources include [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease).

---

## Model Training

- The model is trained using `Train_plant_disease.ipynb`.
- Uses convolutional layers, pooling layers, and fully connected layers for image classification.
- Training parameters (epochs, batch size, augmentation) can be adjusted in the notebook.
- After training, the model weights are saved as `trained.h5` and `trained.keras`.

---

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Predict Disease from an Image

```bash
python main.py --image path/to/leaf_image.jpg
```
- The script loads the pre-trained model and outputs the predicted disease class.

### 3. Train Your Own Model

- Use `Train_plant_disease.ipynb` notebook.
- Adjust parameters as needed.
- Save the trained model for future predictions.

### 4. Test the Model

- Use `Test_Disease.ipynb` notebook to evaluate prediction accuracy and visualize results.

---

## Requirements

- Python 3.7+
- Keras
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- (See `requirements.txt` for full list)

---

## Evaluation

- The model's performance is evaluated using accuracy, precision, recall, and confusion matrix.
- Test results are available in `Test_Disease.ipynb`.
- For best results, ensure dataset is balanced across all disease classes.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes.
4. Submit a pull request.

Feel free to open issues for bug reports or feature requests.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Author

**Arun Sundar**

For questions or collaborations, connect via [GitHub](https://github.com/ArunSundar-1805).

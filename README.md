# Handwritten Character Recognition

## Overview

This project focuses on developing a robust system for recognizing handwritten alphanumeric characters using the EMNIST Balanced Dataset. The system employs a Convolutional Neural Network (CNN) to achieve high accuracy in character prediction. A user-friendly interface allows users to draw characters on a canvas, which are then processed and classified by the trained model in real time. The model achieves approximately 90% accuracy on the validation set, demonstrating its effectiveness in handling diverse handwriting styles.

## Dataset Overview

The EMNIST (Extended MNIST) dataset is an extension of the classic MNIST dataset, incorporating handwritten digits and letters. Derived from the NIST Special Database 19, EMNIST provides a more comprehensive and challenging benchmark for character recognition tasks. Key features of the dataset include:

- **Balanced Classes**: The EMNIST Balanced Dataset consists of 47 classes (10 digits, 26 uppercase letters, and 11 lowercase letters), with each class containing an equal number of samples to ensure fairness in training.
- **Preprocessing**: The dataset includes grayscale images of size 28x28 pixels, normalized and centered to match the format of the original MNIST dataset. The images undergo rotation and flipping to correct orientation discrepancies.
- **Training and Testing Splits**: The dataset is divided into training and testing sets, with a validation subset included for iterative model evaluation.

The dataset's diversity in handwriting styles and balanced class distribution make it an ideal choice for training models to recognize handwritten characters accurately.

## EMNIST Dataset Analysis Notebook

For a deeper understanding of the EMNIST dataset's features and structure, refer to the included notebook titled **"EMNIST_Balanced_Dataset_Analysis_using_Exploratory_Data_Analysis.ipynb"** available in the `notebooks` folder. This notebook provides a comprehensive Exploratory Data Analysis (EDA) of the dataset, covering:

- Dataset loading and basic inspection
- Visualization of sample images
- Analysis of class distribution
- Normalization of pixel values
- Key insights into the dataset's characteristics

The notebook is designed to help users familiarize themselves with the dataset before proceeding with model training or further analysis. It includes detailed code snippets, visualizations, and explanations to ensure clarity and ease of use.

## Model Architecture and Overview

The model is a Convolutional Neural Network (CNN) designed to handle the 28x28 grayscale images from the EMNIST dataset. The architecture consists of the following layers:

1. **Convolutional Blocks**:
   - Three sequential blocks, each comprising:
     - A 2D convolutional layer with kernel size 3x3 and padding.
     - Batch normalization for stable training.
     - ReLU activation for non-linearity.
     - Max pooling for dimensionality reduction.

2. **Fully Connected Layers**:
   - A flattening layer to convert feature maps into a vector.
   - Two dense layers with ReLU activation and dropout for regularization.
   - An output layer with softmax activation for multi-class classification.

**Training Details**:
- **Optimizer**: Adam with a learning rate of 0.001.
- **Loss Function**: Cross-entropy loss.
- **Learning Rate Scheduling**: ReduceLROnPlateau to adaptively adjust the learning rate based on validation loss.
- **Epochs**: 20, with batch size 64.

The model achieves approximately 90% accuracy on the validation set, demonstrating its capability to generalize well to unseen data.

## Evaluation Metrics

The model's performance is evaluated using the following metrics:
- **Accuracy**: The percentage of correctly classified characters on the validation set (~90%).
- **Loss**: Training and validation loss are tracked to monitor convergence and detect overfitting.
- **Confusion Matrix**: Used to identify misclassifications and assess performance per class.

The training process includes plotting loss and accuracy curves to visualize model performance over epochs, ensuring transparency in evaluation.

## How to Use the Application

1. **Prerequisites**:
   - Python 3.10 or later.
   - Required libraries: PyTorch, TorchVision, NumPy, Matplotlib.
   - GPU support (optional but recommended for faster training).

2. **Steps**:
   - Clone the repository and install dependencies.
   - Load the dataset using the provided code, ensuring the correct transformations are applied.
   - Train the model by running the Jupyter Notebook (`notebooks/CNN_characters_model.ipynb`).
   - Use the saved model (`model/balanced/cnn_model.pth`) for inference in your application.
   - Deploy the canvas interface for real-time character drawing and prediction.

3. **Inference**:
   - Preprocess user-drawn images to match the model's input requirements (28x28 grayscale, normalized).
   - Pass the image through the trained model to obtain the predicted character.

## Limitations

- **Character Similarity**: The model may struggle with characters that are visually similar (e.g., 'I', 'L', '1').
- **Dataset Bias**: The EMNIST dataset primarily contains samples from English writers, which may limit performance on handwriting styles from other languages or regions.
- **Real-World Variability**: Handwritten characters drawn by users may vary significantly from the training data, affecting prediction accuracy.
- **Computational Requirements**: Training the model requires significant computational resources, especially without GPU acceleration.

## Getting Started  

Follow these steps to set up and run the Handwritten Character Recognition project locally.  

### 1. Clone the Repository  
First, clone the project repository to your local machine:  
```bash
git clone https://github.com/barkiayoub/handwritten-character-recognition.git
cd handwritten-character-recognition
```  

### 2. Install Dependencies  
Create a virtual environment (recommended) and install the required Python packages:  
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```  

### 3. Run the Application  
Start the Flask server to launch the application:  
```bash
python app.py
```  
The application should now be running at `http://127.0.0.1:5000/`. Open this URL in a web browser to access the character recognition interface.  

**Note:** Ensure the trained model (`model/balanced/cnn_model.pth`) is placed in the correct directory before running the application.

## Contributors

This project was developed as part of an academic initiative. Contributions are welcome, and users are encouraged to submit issues or pull requests for improvements.

---

For questions or feedback, please contact ayoubbarki17@gmail.com.
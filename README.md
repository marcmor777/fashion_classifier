# Fashion Classifier Using TensorFlow y Keras

This project uses a sequential neural network, built with Keras and TensorFlow, to classify 10 different types of clothing from the Fashion MNIST dataset. The model achieves an **accuracy of approximately 88%** on the test dataset.

---

### Result Visualization

![Training and Validation Graphs](images/training_history.png)
*Graphs showing the evolution of accuracy and loss during training.*

---
### Technologies Used
- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib

---
### Installation

To run this project, clone the repository and create a virtual environment. 

```bash
# Clone the repo
git clone [https://github.com/tu-usuario/fashion-classifier.git](https://github.com/tu-usuario/fashion-classifier.git)
cd fashion-classifier

# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # in Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---
### Usage

To train model execute the `train.py` script from the `src` folder.

```bash
python src/train.py
```

The script will execute the following steps:
- **Data Loading and Preprocessing**: Loads and preprocesses the Fashion MNIST dataset.
- **Model Building**: Constructs and compiles the neural network architecture.
- **Training**: Trains the model for 20 epochs.
- **Evaluation**: Evaluates the model's performance on the test set.
- **Saving**: Saves the trained model to the `models/` folder and performance graphs to `images/`.

### Model Architecture 

The model is a sequential neural network with the following layers:
- **Flattening Layer:** Flattens the 28x28 images into a 784-pixel vector.
- **Hidden Dense Layer:** 128 neurons with ReLU activation.
- **Dropout Layer:** 0.3 rate for regularization and overfitting prevention.
- **Output Dense Layer:** 10 neurons (one per class) with Softmax activation for classification.

---
### Conclusion and Future Improvements

This project successfully fulfills its objective of building a functional image classifier using the fundamentals of neural networks and Keras, achieving a remarkable accuracy of **88%**. This validates that the entire workflow—from preprocessing to training and evaluation—is robust.

The main area for improvement lies in the model architecture. The current dense neural network, while effective, does not leverage the spatial information inherent in images. By flattening the 28x28 matrix into a vector, the model loses the notion of which pixels are close to others.

To take this project to the next level, the clear strategy is as follows:
1.  **Implement a Convolutional Neural Network (CNN):** Replace the initial dense layers with convolutional (`Conv2D`) and pooling (`MaxPooling2D`) layers. This would allow the model to learn hierarchies of visual features, from simple edges in the early layers to complex shapes in the deeper ones.
2.  **Hyperparameter Optimization with KerasTuner:** Conduct a systematic search for the best hyperparameters for the CNN architecture (number of filters, kernel size, etc.) to extract maximum performance.

With these changes, the accuracy on the test set is expected to exceed 95%, approaching the state-of-the-art for the Fashion MNIST dataset.
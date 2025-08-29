import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os



def load_and_preprocess_data():
    """Loads Fashion MNIST dataset from Keras
    Input:
        None
    Output: 
        (train_images, train_labels), (test_images, test_labels), class_names
    
    """
    
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    # Normalize data in range [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                   
    return (train_images, train_labels), (test_images, test_labels), class_names

def build_model():
    """
    Builds a Keras Sequential model for image classification.
    Input:
        None
    Output:
        model (keras.Model): Compiled Keras model.
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

def plot_training_history(history, save_path="images"):
    """
    Plots the training history (accuracy and loss) and saves the plot as an image.
    Input:
        history (keras.callbacks.History): History object returned by model.fit().
        save_path (str): Directory to save the plot.
    Output:
        None
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(12, 5))

    # Graficar precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Graficar pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    print(f"\nTraining Graphs saved to '{save_path}/training_history.png'")

if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)

    # Load Data
    (train_images, train_labels), (test_images, test_labels), class_names = load_and_preprocess_data()

    # Build Model
    model = build_model()

    # Compile Model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    # Train Model
    history = model.fit(train_images, train_labels, 
                        epochs=20, 
                        validation_split=0.2,
                        verbose=2)

    # Evaluar modelo
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nOverall Accuracy in Test Set: {test_acc:.4f}')

    # Guardar gráficas de resultados
    plot_training_history(history)

    # Guardar el modelo entrenado
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/fashion_classifier_model.pickle')
    print("\nModel saved to 'models/fashion_classifier_model.pickle'")
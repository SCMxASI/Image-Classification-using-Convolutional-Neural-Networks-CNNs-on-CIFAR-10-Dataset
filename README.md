# Image-Classification-using-Convolutional-Neural-Networks-CNNs-on-CIFAR-10-Dataset
This project demonstrates the use of Convolutional Neural Networks (CNNs) for image classification on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal of this project is to train a CNN model to classify these images into their respective classes.


NOTE: *I don't attach any Libraries which are essential to run this program. Please make sure all the libraries are downloaded before running this code.*

->NEEDED LIBRARIES:
******************
1.NumPy (np): A library for efficient numerical computation in Python.

2.TensorFlow (tf): An open-source machine learning library developed by Google.

3.Keras: A high-level neural networks API, capable of running on top of TensorFlow, CNTK, or Theano.

4.Scikit-learn: A machine learning library for Python, used for tasks such as data preprocessing, feature selection, and model evaluation.

5.Matplotlib (plt): A plotting library for creating static, animated, and interactive visualizations in Python.

6.Seaborn (sns): A visualization library based on Matplotlib, used for creating informative and attractive statistical graphics.

********************************************************************************************************************************

->Dataset:
**********
The CIFAR-10 dataset is loaded using the keras.datasets.cifar10.load_data() function. The dataset is split into training and testing sets, with 50,000 images for training and 10,000 images for testing.

->Data Preprocessing:
*********************
The images are normalized by dividing the pixel values by 255, which scales the values between 0 and 1. This is done to improve the stability of the neural network.

->Model Architecture:
*********************
The CNN model is defined using the keras.Sequential API. The model consists of the following layers:

Conv2D Layer: This layer applies 32 convolutional filters to the input image, with a kernel size of 3x3 and ReLU activation.
MaxPooling2D Layer: This layer applies max pooling to the output of the convolutional layer, with a pool size of 2x2.
Flatten Layer: This layer flattens the output of the max pooling layer into a 1D array.
Dense Layer: This layer applies a dense layer with 64 units and ReLU activation.
Output Layer: This layer applies a dense layer with 10 units and softmax activation, which outputs the probabilities of each class.

->Model Compilation:
********************
The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function. The accuracy metric is also tracked during training.

->Training:
***********
The model is trained on the training set for 10 epochs, with a batch size of 32. The validation set is used to evaluate the model's performance during training.

->Evaluation:
*************
The model's performance is evaluated on the test set, and the accuracy is printed to the console.

->Visualization:
****************
The training and validation accuracy are plotted using Matplotlib, which shows the model's performance during training.

->Prediction:
*************
The model is used to make predictions on the test set, and the predicted classes are obtained by taking the argmax of the output probabilities.

->Displaying Predictions:
*************************
The images with their predicted classes are displayed using Matplotlib, which shows the model's performance on the test set.

->Code:
*******
The code is written in Python using the Keras API. The code is well-structured and follows best practices for readability and maintainability.

->Requirements:
***************
Python 3.6+
Keras 2.3.1+
TensorFlow 2.2.0+
Matplotlib 3.2.1+
NumPy 1.19.2+
Scikit-learn 0.23.2+

->Usage:
*********
Clone the repository: gh repo clone SCMxASI/Image-Classification-using-Convolutional-Neural-Networks-CNNs-on-CIFAR-10-Dataset
Install the requirements: pip install -r requirements.txt
Run the code: python main.py
Commit Message Guidelines
Use the present tense ("Add feature" instead of "Added feature")
Use the imperative mood ("Fix bug" instead of "Fixes bug")
Limit the first line to 72 characters or less
Reference issues and pull requests liberally after the first line
API Documentation Guidelines
Use clear and concise language
Use proper grammar and spelling
Use consistent formatting and indentation
Include examples and code snippets where applicable
*****************************************************************************************************************************************

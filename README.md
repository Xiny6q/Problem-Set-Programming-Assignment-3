Download link :https://programming.engineering/product/problem-set-programming-assignment-3/

# Problem-Set-Programming-Assignment-3
Problem Set + Programming Assignment 3
Loss, Gradients, and PyTorch

1.1 Written Portion – answer within PDF

Preliminaries: Recall our discussion of loss functions from class. For a multi-class clas-sification problem, given a prediction model f, a common loss function is the Categorical Cross Entropy loss, which is calculated as follows:

L(xi, yi) = −yi · log f(xi)

Here, yi is a one-hot vector (i.e. a vector of length m, where m is the number of classes) with an entry of 1 in the position corresponding to the true class and 0 elsewhere. Simi-larly, the model output f(xi) is a vector, containing the probabilities that the input belongs to each of the classes.

Q1. Consider a logistic regression (yˆi = σ(w · xi + b)) with two classes (0 and 1). Given the input example, xi = [6, 4, 2, 3, 1], model weights w = [0.5, 0.1, −0.8, 0.9, 0.0], bias term, b = 0.05 and the true label encoded as yi = [0, 1], compute the cross entropy loss and show every step of your calculation. (Hint: a logistic regression gives you the probability

of the ‘1’ class.)

(4)

Q2. The Rectified Linear Unit (ReLU) activation function is defined as:

ReLU(x) = x + |x| = x if x > 0,

0 otherwise

Now consider the neural network in the figure below, with 3 possible classes instead. Assume that the hidden layer uses the ReLU activation function instead of the sigmoid function discussed in class. The final layer uses the sigmoid (logistic function) activation function, followed by a Softmax operation to ensure that the outputs add up to 1.



Weights

Bias

v1

[0.5, −0.1, −0.2, 0.1, −0.4]

0.02

v2

[−0.9, 0.7, 0.7, −0.5, 0.2]

−0.01

w1

[0.1, 0.6]

0.0

w2

[0.8, 0.4]

0.05

w3

[0.7, −0.2]

0.04

Figure 1: A simple neural network and its parameters

For xi = [5, 7, 4, 3, 2], and the true class label 2 (where the possible classes are 0, 1, and 2), use the model from the diagram, and the given weights table (including nonzero biases this time) to compute the cross entropy loss. You may use NumPy/SciPy operations (but not the loss function methods from these packages) to perform the calculations for this question – if doing so, paste a screenshot of your function(s) and the final result into your

written submission. You may find the SciPy expit function useful: [link].

(6)

Q3. Using a computation graph (of the kind we discussed in class), compute the gradient of the cross entropy loss with respect to the model parameters in the above network. Re-peatedly split the loss function by operators like we did in class, and ignore all bias terms. You do not need to split vector/matrix operators into elementwise/row-wise operations. For this question alone, you may scan a hand-drawn figure. Any writing must be perfectly legible. Finally, use NumPy to compute the value of the gradient of the loss function with respect to all parameters (w1, w2, w3, v1, v2) for the given input and true label. Note that the log operator refers to the natural logarithm (base exp). Report your final values and a

screenshot of your code. HINT: [Derivative of Softmax]

(5+5)

Fashion-MNIST Classification

2.1 Implement and train a Neural Network

(30)

Relevant file: fashionmnist.py

In this section, you will be working with a dataset called Fashion-MNIST. Your task is to design and train a neural network that classifies each image of the dataset correctly. Data is downloaded directly from within the script (using PyTorch). The actual architecture of the neural network is up to you, and convolutional layers are optional.

Fashion-MNIST contains grayscale images of 28 x 28 pixels representing images of cloth-ing. The dataset has 60000 training images, and 10000 testing images, and each image comes with an associated label (e.g. t-shirt, coat, bag, etc.). There are 10 classes, just like the MNIST handwritten digits dataset, so that it may serve as a direct drop-in replace-ment to test neural networks. Read the full details about this dataset at the repository.


The starter code for this part of the assignment is called “fashionmnist.py”. The skeleton code serves as a general guide. There are some sections without any guidelines where you will be expected to research and experiment with various techniques to find a good approach. You must use PyTorch in this section, which is well-documented online.

Your accuracy on the testing dataset must be greater or equal to 80%.

2.2 Written Portion – answer within PDF

(The points from Q4 and Q5 will be added to the programming portion of the grade for this assignment.)

Q4. Submit two figures: A figure containing an image that is classified incorrectly by your model. Include a clear label in this figure that indicates the predicted class from your model and the true class the image belongs to (both human-readable labels, not just the class number). The second figure should be a single image classified correctly by your

model and its corresponding class label.

(3)

Q5. Submit a plot showing your training loss over time.

(2)

Q6. Assume both the testing and training datasets have twice as many coat images as

shirt images, and answer the following questions:

(6)

Would accuracy still be a good metric?

How would you modify your accuracy metric to account for data imbalance?

What other metrics can you use to evaluate model performance?

Q7. Calculate the total number of tunable parameters your model has. Show your work,

and don’t forget the bias terms. Do not use code output to answer this question.

(4)


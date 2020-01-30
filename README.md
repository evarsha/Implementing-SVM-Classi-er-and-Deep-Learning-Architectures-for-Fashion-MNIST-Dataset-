# Implementing-SVM-Classi-er-and-Deep-Learning-Architectures-for-Fashion-MNIST-Dataset-

Implemented the SVM Classifier and Deep Learning Architectures for classifying the
Fashion-MNIST dataset. It consists of 60,000 28 x 28 pixel grayscale images in its training set which
are categorized into 10 classes, and the testing set consists of 10,000 images.

Each training and test example is assigned to one of the following labels: 0=T-shirt/top, 1=Trouser,
2=Pullover, 3=Dress, 4=Coat, 5=Sandal, 6=Shirt, 7=Sneaker, 8=Bag, 9=Ankle boot


I have implemented the flattened data for all the architectures except for the ResNet50 architecture.
 
 
 I have implemented Linear, Kernel(Polynomials defined in the SVM section) and RBF SVM by applying PCA and LDA for Dimensionality Reduction for all types.
 
 
 I have also implemented SVM classifier without Dimensionality Reduction for computing its accuracy
and computative cost(time).
For Deep Learning architectures I have implemented the traditional CNN using TensorFlow and Numpy
libraries and ResNet50 Architecture using Keras in my second model.

The results obtained, their computation time, trade-os and architectures have been described.

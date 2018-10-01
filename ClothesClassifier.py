# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__) #test to make sure tensorflow properly installed

"""This guide uses the Fashion MNIST dataset which contains 70,000 grayscale images 
in 10 categories. The images show individual articles of clothing at low resolution (28 by 28 pixels)
We will use 60,000 images to train the network and 10,000 images to evaluate how accurately the 
network learned to classify images."""

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

"""The labels are an array of integers, ranging from 0 to 9. 
These correspond to the class of clothing the image represents:
0 = Shirts, 1 = Pants, 2 = Pullover, etc"""
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape) #should print (60000, 28, 28) for 60k training images that are 28x28 pixels
print(len(train_labels)) #prints number of labels, which is 60k

#prepare data by converting to float
train_images = train_images / 255.0

test_images = test_images / 255.0

"""verifies data is in correct format - uncomment to see"""
"""plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])"""

#plt.show()

"""The first layer in this network, tf.keras.layers.Flatten, transforms the format 
of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels.
After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense 
layers. These are densely-connected, or fully-connected, neural layers. 
The first Dense layer has 128 nodes. 
The second (and last) layer is a 10-node softmax layer. This returns an array of 10 
probability scores that sum to 1. Each node contains a score that indicates the 
probability that the current image belongs to one of the 10 classes."""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

"""Loss function —This measures how accurate the model is during training. 
We want to minimize this function to "steer" the model in the right direction.

Optimizer — This is how the model is updated based on the data it sees and its loss function.

Metrics — Used to monitor the training and testing steps. The following example uses accuracy, 
the fraction of the images that are correctly classified"""

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""Fit the training data for 5 epochs"""
model.fit(train_images, train_labels, epochs=5)

"""Evaluate Accuraxcy"""
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
#getting accuracy of about 87.5%. This is likely caused by overfitting

"""Making Predictions: A prediction is an array of 10 numbers. 
These describe the "confidence" of the model that the image corresponds to 
each of the 10 different articles of clothing. We can see which label has 
the highest confidence value"""
predictions = model.predict(test_images)

#We can see which label has the highest confidence value
print(np.argmax(predictions[0]))
#will return 9, meaning it believes the first item of clothing is an Ankle Boot


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# Grab an image from the test dataset
img = test_images[0]
print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)

#Now predict the image
predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])
#Returns a 9, correctly predicting image 0 is an ankle boot
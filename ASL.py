# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

"""## Question 1 - Gesture Recognition using Convolutional Neural Networks
---
American Sign Language (ASL) is a complete, complex language that employs signs made by moving the hands combined with facial expressions and postures of the body. It is the primary language of many North Americans who are deaf and is one of several communication options used by people who are deaf or hard-of-hearing.

The hand gestures representing English alphabet are shown below. In this question, you will focus on classifying these hand gesture images using convolutional neural networks. Specifically, given an image of a hand showing one of the letters, we want to detect which letter is being represented.

Run the following code cell to download the training and test data. It might take a while to download the zip file and extract it.
"""

import os
import shutil
from sklearn.preprocessing import LabelBinarizer
from matplotlib import image
import string
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import io
import zipfile
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

file_id = '1jDDjJzhZ7XV5DMRAEg23ymrz-IjnpGIO'
downloaded = drive.CreateFile({'id': file_id})
downloaded.GetContentFile('data.zip')

zip_ref = zipfile.ZipFile("data.zip", 'r')
with zipfile.ZipFile('./data.zip', 'r') as zip_ref:
    zip_ref.extractall('./')
zip_ref.close()

"""###Part A) Understanding and Processing the Data

Now that you downloaded the data, you see two folders containing training and test data. Complete the following steps:

1) read in the training and test data. Examine the data folders carefully to see how file names and folder names represent different labels (29 labels in total) in the datasets.

2) rescale the pixel values of the training and test images from [0,255] to [0,1].

3) make sure that all of your imges are of size $200\times 200$. If not, scale them appropriately.

4) Ensure that your target values (classes) are stored appropriately. You must have 29 classes for 'a-z', 'del', 'nothing', and 'space'.
"""

dir = "asl_alphabet_train/asl_alphabet_train/"
val_dir = "asl_alphabet_train/asl_alphabet_val/"
test_dir = "asl_alphabet_train/asl_alphabet_test/"

folders = os.listdir(dir)
os.mkdir(val_dir)
os.mkdir(test_dir)
for d in folders:
  os.mkdir(val_dir + d)
  os.mkdir(test_dir + d)

for i, label in enumerate(folders):
  all_files = os.listdir(dir + label)
  np.random.shuffle(all_files)

  if i == 0:
    num_train = np.int(.6 * len(all_files))
    num_val = np.int(.2 * len(all_files))

  train = all_files[:num_train]
  val = all_files[num_train: num_train + num_val]
  test = all_files[num_train + num_val:]

  for img in val:
    shutil.move(dir + label + "/" + img, val_dir + label + "/")
  for img in test:
    shutil.move(dir + label + "/" + img, test_dir + label + "/")

with tf.device('/device:GPU:0'):
  train_datagen = ImageDataGenerator(rescale=1./255)
  val_datagen = ImageDataGenerator(rescale = 1./255)
  test_datagen = ImageDataGenerator(rescale = 1./255)

  train_generator = train_datagen.flow_from_directory(
      dir,
      target_size=(200, 200),
      batch_size = 256,
      class_mode = 'categorical',
  )

  validation_generator = val_datagen.flow_from_directory(
      val_dir,
      target_size=(200, 200),
      batch_size = 256,
      class_mode = 'categorical',
  )

  test_generator = test_datagen.flow_from_directory(
      test_dir,
      target_size=(200, 200),
      batch_size = 256,
      class_mode = 'categorical',
  )

"""### Part B) Building a Convolutional Neural Network
For this assignment, we are not going to give you any starter code. You will be writing a convolutional neural network from scratch. You are welcome to use any code from previous class exercises, section handouts, and lectures. You should also write your own code.

You may use the TensorFlow documentation freely. You might also find online tutorials helpful. However, all code that you submit must be your own.

Make sure that your code is vectorized, and does not contain obvious inefficiencies (for example, unnecessary for loops). Ensure enough comments are included in the code so that your TA can understand what you are doing. It is your responsibility to show that you understand what you write.

Follow the steps below to show your work.

#### Step 1) Building the Network
Build a convolutional neural network model that takes the ($200\times 200$ RGB) image as input, and predicts the letter. Explain your choice of the architecture: how many layers did you choose? What types of layers did you use? Were they fully-connected or convolutional? What about other decisions like pooling layers, activation functions, number of channels / hidden units?
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

with tf.device('/device:GPU:0'):
  model = Sequential()
  model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
  model.add(MaxPooling2D(pool_size=(20, 20)))
  model.add(Dropout(0.4))
  model.add(Conv2D(filters=128, kernel_size=3, activation="relu"))
  model.add(MaxPooling2D(pool_size=(4, 4)))
  model.add(Dropout(0.4))

  model.add(Flatten())
  model.add(Dense(256, activation="relu"))
  model.add(Dropout(0.4))
  model.add(Dense(128, activation="relu"))
  model.add(Dropout(0.4))

  model.add(Dense(29, activation="softmax"))

  opt = Adam(learning_rate=0.0005)
  model.compile(
      optimizer=opt,
      loss="categorical_crossentropy",
      metrics=["accuracy"]
  )

"""I chose two layers of convolution and then three fully connected layers. I added many dropout layers in order to have a regularizing effect. I really wanted to avoid overfitting, and without the dropout layers I noticed there was a lot of overfitting to the test data compared to the validation data. This caused the overall training accuracy/loss to be worse, however there was a much smaller gap between the training accuracy/loss and the validation accuracy/loss, meaning the model was more robust to different data, which in my mind was worth the trade off. In the end, this model would simply take longer to train to completion but would be more robust overall. Since the imput size is 200x200 I thought that an initial pool size of 20, 20 was suitable since each dimension is 10x less than the overall size of the image. This is then scaled down to 4x4, which extracts more features from the ones already extracted by the previous layer. I did not want to add many more hidden layers to my fully connected layers because it would slow down the already slow training speed even more, and decided that two layers were enough.

#### Step 2) Training the Network
Write code that trains your neural network given the training data. Your training code should make it easy to tweak the usual hyperparameters, like batch size, learning rate, and the model object itself. Make sure that you are checkpointing your models from time to time (the frequency is up to you). Explain your choice of loss function and optimizer.

Plot the training curve as well.
"""

checkpoint_dir = "checkpoint/"
os.mkdir(checkpoint_dir)
checkpoint_filepath = checkpoint_dir
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

with tf.device('/device:GPU:0'):
  history = model.fit_generator(
      train_generator,
      steps_per_epoch = 2000 // 256,
      validation_data = validation_generator,
      validation_steps = 2000 // 256,
      epochs = 200,
      callbacks = [model_checkpoint_callback]
  )

with tf.device('/device:GPU:0'):
  scores = model.evaluate_generator(test_generator, 200)
print("Test set accuracy: ", scores[1])

def plot_losses(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'])
    plt.show()

def plot_accuracies(hist):
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'])
    plt.show()

plot_losses(history)
plot_accuracies(history)

"""I chose the adam optimizer after doing my own reseach and finding "Adam combines the best properties of the AdaGrad and RMSProp" to produce the most stable models. Additionally, I chose the categorical_crossentropy loss function because we are training many categorical variables which are one-hot encoded (if not one-hot encoded I would use sparse_categorical_crossentropy, according to documentation.

#### Step 3) Overfit to A Small Dataset

One way to sanity check our neural network model and training code is to check whether the model is capable of overfitting or memorizing a small dataset. A properly constructed CNN with correct training code should be able to memorize the answers to a small number of images quickly.

Construct a small dataset (e.g. sample from the training data). Then show that your model and training code is capable of memorizing the labels of this small data set.

With a large batch size (e.g. the entire small dataset) and a learning rate that is not too high, you should be able to obtain a 100% training accuracy on that small dataset relatively quickly.
"""

label_map = dict(zip(string.ascii_lowercase, range(0, 26)))
label_map['del'] = 26
label_map['nothing'] = 27
label_map['space'] = 28
directory_types = list(label_map.keys())
for i in range(26):
  directory_types[i] = directory_types[i].upper()

def load_train_asl_data(directory, sample_size):
  X = np.empty((sample_size, 200, 200, 3), dtype=np.uint8)
  y = np.empty((sample_size,), dtype=np.uint8)
  for dir_type in directory_types:
    i = 0
    for filename in os.listdir(directory+dir_type)[:sample_size]:
      if filename.endswith(".jpg"):
        img_dir = os.path.join(directory+dir_type, filename)
        X[i] = image.imread(img_dir)
        y[i] = label_map[dir_type.lower()]
        i+=1
  return X, y

x_train, y_train = load_train_asl_data(dir, 300)
x_train = x_train/255.

model2 = tf.keras.models.clone_model(model)
opt = Adam()
model2.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

from sklearn.model_selection import train_test_split

x_smol, _, y_smol, _ = train_test_split(x_train, y_train, test_size = 0.90, random_state=144)
y_smol.shape

history2 = model2.fit(
    x_smol,
    y_smol,
    batch_size = 256,
    epochs = 50
)
"""

#hyperparameter list
#batch size: 256 -> 128
#dropout: 0.4 -> 0.25
#learning rate: 0.001 -> 0.0015
#going to optimize learning rate with values 0.0005, 0.001, 0.0015, 0.002

"""Learning rate = 0.003
loss: 1.7671 - accuracy: 0.3990 - val_loss: 1.4133 - val_accuracy: 0.5368
Learning rate = 0.002
Learning rate = 0.001
Test set accuracy:  0.6508620977401733
Learning rate 0.0005
Based on the tests, I believe that using a learning rate of 0.001 is the best because it gives the least disparity between the training set loss and accuracy and the validation loss and accuracy while still being able to improve its performance on the dataset. Based on the graphs, not enough epochs have been included in order to fully train the network. However, due to time constraints with each training taking upwards of 30 minutes, I decided that it was acceptable to leave the models trained at 200 epochs.

## Question 2 - Transfer Learning
---
For many image classification tasks, it is generally not a good idea to train a very large deep neural network model from scratch due to the enormous compute requirements and lack of sufficient amounts of training data.

One of the better options is to try using an existing model that performs a similar task to the one you need to solve. This method of utilizing a pre-trained network for other similar tasks is broadly termed Transfer Learning. In this assignment, we will use Transfer Learning to extract features from the hand gesture images. Then, train a smaller network to use these features as input and classify the hand gestures.

As you have learned from the CNN lecture, convolution layers extract various features from the images which get utilized by the fully connected layers for correct classification.

Keras even has pretrained models built in for this purpose.

#### Keras Pretrained Models
        Xception
        VGG16
        VGG19
        ResNet, ResNetV2, ResNeXt
        InceptionV3
        InceptionResNetV2
        MobileNet
        MobileNetV2
        DenseNet
        NASNet

Usually one uses the layers of the pretrained model up to some point, and then creates some fully connected layers to learn the desired recognition task. The earlier layers are "frozen", and only the later layers need to be trained. We'll use VGG16, which was trained to recognize 1000 objects in ImageNet. What we're doing here for our classifier may be akin to killing a fly with a shotgun, but the same process can be used to recognize objects the original network couldn't (i.e., you could use this technique to train your computer to recognize family and friends).
"""

# Some stuff we'll need...
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

"""Creating this pretrained network is a one line command. Notice we specified that the "top" should not be included. We aren't classifying 1000 different categories like ImageNet, so we don't include that layer. We'll add our own layer more suited to the task at hand.

We choose 224 as our image dimension because the pretrained VGG16 was trained using the ImageNet dataset which has images of this dimension.
"""

# Import the VGG16 trained neural network model, minus it's last (top) neuron layer.
base_model = VGG16(weights = 'imagenet',
                   include_top = False,
                   input_shape = (200, 200, 3),
                   pooling = None)

"""Let's take a look at this pretrained model:"""

base_model.summary()

"""Please do realize, this may be overkill for our toy recognition task. One could use this network with some layers (as we're about to add) to recognize 100 dog breeds or to recognize all your friends. If you wanted to recognize 100 dog breeds, you would use a final 100 neuron softmax for the final layer. We'll need a final softmax layer as before. First let's freeze all these pretrained weights. They are fine as they are."""

# This freezes the weights of our VGG16 pretrained model.
for layer in base_model.layers:
    layer.trainable = False

"""### Part A) Building the Classifier
Now let's just add a flatten layer, a trainable dense layer, and a final softmax layer to the network to complete the classifier model for our gesture recognition task. Use Keras' functional approach to building a network.
"""

# Now add layers to our pre-trained base model and add classification layers on top of it
x = base_model.output
x = Flatten()(x)
x = Dense(200, activation="relu")(x)
x = Dropout(0.4)(x) #added dropout layer for regularization
predic = Dense(29, activation="softmax")(x)

# And now put this all together to create our new model.
model3 = Model(inputs = base_model.input, outputs = predic)
model3.summary()

"""### Part B) Initializing Training Parameters

Compile the model using an appropriate loss function and optimizer.
"""

# Compile the model.
model3.compile(loss="categorical_crossentropy",
              optimizer=Adam(lr = 0.001), #change to adam optimizer
              metrics=["acc"])

"""### Part C) Training the Model

Train your new network, including any hyperparameter tuning. Plot the training curve of your best model only.

As you can see here in the Keras docs:

https://keras.io/api/applications/vgg/#vgg16-function

that we are required to preprocess our image data in a specific way to use this pretrained model, so let's go ahead and do that first.
"""

# Preprocess your input image data
train_datagen2 = ImageDataGenerator(rescale=1./255, validation_split=0.2, preprocessing_function=preprocess_input)

train_generator2 = train_datagen2.flow_from_directory(
    dir,
    target_size=(200, 200),
    batch_size = 256,
    class_mode = 'categorical',
    subset = 'training'
)

validation_generator2 = train_datagen2.flow_from_directory(
    dir,
    target_size=(200, 200),
    batch_size = 256,
    class_mode = 'categorical',
    subset = 'validation'
)

# Train the model
epochs = 10
history3 = []
with tf.device('/device:GPU:0'):
  history3 = model3.fit_generator(train_generator2,
                                  validation_data = validation_generator2,
                                  epochs = epochs,
                                  steps_per_epoch = 163,
                                  validation_steps = 40,
                                  verbose=1)

def plot_losses2(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'])
    plt.show()

def plot_accuracies2(hist):
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'])
    plt.show()

# Plot the training curve
plot_losses2(history3)
plot_accuracies2(history3)

"""Adagrad optimizer
loss: 0.5271 - acc: 0.9385 - val_loss: 1.0520 - val_acc: 0.7586

Adam optimizer with dropout
loss: 0.1404 - acc: 0.9549 - val_loss: 0.4517 - val_acc: 0.8652
### Part D) Your Best Classifier

Add on your own last layers to the pretrained model and train it on the training data (in the previous parts you could have only one flatten layer and one dense layer to do the classification). You can increase (or decrease) the number of nodes per layer, increase (or decrease) the number of layers, and add dropout if your model is overfitting, change the hyperparameters, change your optimizer, etc. Try to get the validation accuracy higher than what the previous transfer learning model was able to obtain, and try to minimize the amount of overfitting.

Plot the classification accuracy for each epoch. Report the best test accuracy your model was able to achieve.
"""

with tf.device('/device:GPU:0'):
  scores = model3.evaluate_generator(test_generator, 200)
print("Test set accuracy: ", scores[1])

"""The best test accuracy the model could achieve was 0.9521264433860779. """

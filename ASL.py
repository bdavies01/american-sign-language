# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

# Gesture Recognition using Convolutional Neural Networks

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

"""Processing the Data
The next block performs the following steps:
1) Read in the training and test data.
2) Rescales the pixel values of the training and test images from [0,255] to [0,1].
3) Makes sure that all of the images are of size 200x200. If not, scale them appropriately.
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

"""
Building the Convolutional Neural Network
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

"""
Model Notes
I chose two layers of convolution and then three fully connected layers.
I added many dropout layers in order to have a regularizing effect.
I really wanted to avoid overfitting, and without the dropout layers I noticed
there was a lot of overfitting to the test data compared to the validation data.
This caused the overall training accuracy/loss to be worse, however there was
a much smaller gap between the training accuracy/loss and the validation
accuracy/loss, meaning the model was more robust to different data, which in
my mind was worth the trade off. In the end, this model would simply take
longer to train to completion but would be more robust overall. Since the imput
size is 200x200 I thought that an initial pool size of 20, 20 was suitable since
each dimension is 10x less than the overall size of the image. This is then
scaled down to 4x4, which extracts more features from the ones already extracted
 by the previous layer. I did not want to add many more hidden layers to my
 fully connected layers because it would slow down the already slow training
 speed even more, and decided that two layers were enough.
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

"""
#### Step 3) Overfit to A Small Dataset
To sanity check our neural network model and training code, check whether the
model is capable of overfitting or memorizing a small dataset. A properly
constructed CNN with correct training code should be able to memorize the
answers to a small number of images quickly. Here I construct a small dataset
then show that the model and training code is capable of memorizing the
labels of the small data set.
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
#going to optimize learning rate with values 0.0005, 0.001, 0.0015, 0.002, 0.003

Based on the tests, I believe that using a learning rate of 0.001 is the best
because it gives the least disparity between the training set loss and accuracy
and the validation loss and accuracy while still being able to improve its
performance on the dataset. Based on the graphs, not enough epochs have been
included in order to fully train the network. However, due to time constraints
with each training taking upwards of 30 minutes, I decided that it was
acceptable to leave the models trained at 200 epochs.
"""

##Transfer Learning
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Import the VGG16 trained neural network model, minus it's last (top) neuron layer.
base_model = VGG16(weights = 'imagenet',
                   include_top = False,
                   input_shape = (200, 200, 3),
                   pooling = None)

# This freezes the weights of our VGG16 pretrained model.
for layer in base_model.layers:
    layer.trainable = False

"""Building the Classifier
Add a flatten layer, a trainable dense layer, and a final softmax layer to the
network to complete the classifier model for our gesture recognition task.
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

# Initializing Training Parameters
# Compile the model.
model3.compile(loss="categorical_crossentropy",
              optimizer=Adam(lr = 0.001), #change to adam optimizer
              metrics=["acc"])

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
"""

with tf.device('/device:GPU:0'):
  scores = model3.evaluate_generator(test_generator, 200)
print("Test set accuracy: ", scores[1])

"""The best test accuracy the model could achieve was 0.9521264433860779. """

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import optimizers
from keras.datasets import mnist
from keras.utils import np_utils
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# import warnings
# warnings.filterwarnings('ignore')

'''' 
-------------------------------------------------------------------------
Dataset preparation
-------------------------------------------------------------------------
'''
(X_train, y_train), (X_test, y_test) = mnist.load_data()

class_names = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']

# Display a few images and their label
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
#plt.show()

# number of samples considered:
n_train = 60000
n_test = 10000

# Normalisation 
X_train = X_train[0:n_train,:].astype('float32') / 255.
X_test = X_test[0:n_test,:].astype('float32') / 255.
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

Y_train = np_utils.to_categorical(y_train[0:n_train], 10)
Y_test = np_utils.to_categorical(y_test[0:n_test], 10)


print('shape X train : ', X_train.shape)
print('shape X test : ', X_test.shape)
print('shape Y train : ', Y_train.shape)
print('shape Y test : ', Y_test.shape)


'''' 
-------------------------------------------------------------------------
Model
-------------------------------------------------------------------------
Build a neural network with :

— a convolutional layer with reLU activation, 64 convolutions 5 × 5 (stride 1 × 1). — a max pooling 2×2 (stride 2×2).
— a fully connected layer of 1024 neurons (activation ReLU)
— the output layer
'''
model = Sequential()	

#TODO Complete here!
# First conv2D layer
model.add(Conv2D(32, (5, 5), padding='same', strides=(1,1), input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

# Second conv2D layer
model.add(Conv2D(64, (5, 5), padding='same', strides=(1,1), input_shape=(12,12,32)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

# 这一个很重要，千万不要忘记进行扁平化
model.add(Flatten(input_shape=(4, 4, 64)))

model.add(Dense(units=100, activation='relu', input_dim=1024))
model.add(Dense(units=10, activation='softmax', input_dim= 100))

print(model.summary())

'''' 
-------------------------------------------------------------------------
Training
-------------------------------------------------------------------------
'''

# TODO complete the function compile here:
opt = optimizers.Adam(lr = 0.01)
model.compile(loss='categorical_crossentropy', optimizer= opt, metrics=['accuracy'])


history = model.fit(X_train, Y_train, epochs=6, validation_data = (X_test,Y_test), batch_size=100, verbose = 1)

'''' 
-------------------------------------------------------------------------
Performances
-------------------------------------------------------------------------
'''
# Performance evaluation on the training set
measures_perf_train = model.evaluate(X_train, Y_train)
print('training results : ', measures_perf_train)

# Performance evaluation on the test set
measures_perf_test = model.evaluate(X_test, Y_test)
print('test results : ', measures_perf_test)

'''' 
-------------------------------------------------------------------------
Predictions
-------------------------------------------------------------------------
'''
predictions = model.predict(X_test)

'''' 
-------------------------------------------------------------------------
Visualisation
-------------------------------------------------------------------------
'''
# Plot history
def plot_history(histories, key='acc'):
  plt.figure(figsize=(16,10))
  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')
  plt.xlabel('Epoques')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()
  plt.xlim([0,max(history.epoch)])


plot_history([('performances', history)],
               key='acc')
plt.show()

# display example results
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])  

  plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)  
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
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


pred_labels = np.argmax(predictions, axis=1)
idx_errors = np.where(pred_labels!=y_test[0:n_test])
plt.figure(figsize=(18,12))
K=0
for i in range(12):
#for i in idx_errors[0][0:12]:    
    plt.subplot(4,6,2*K+1)
    plot_image(i, predictions, y_test, X_test)
    plt.subplot(4,6,2*K+2)
    plot_value_array(i, predictions,  y_test)
    plt.xlabel(class_names[y_test[i]])
    K=K+1
plt.show()






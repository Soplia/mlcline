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

X_train = X_train[0:n_train,:].astype('float32') / 255.
X_test = X_test[0:n_test,:].astype('float32') / 255.
#X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
#X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

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
Complete the given script to implement a model with 1 hidden layer of 5 neurons with :
— reLU (rectified Linear Unit) activations in the hidden layer units *
— categorical cross entropy loss. *
— Adam optimizer, with a learning rate of 0.01. *
— batch size of 100 *
'''
model = Sequential()	
model.add(Flatten(input_shape=(28,28)))

#TODO Complete here!
model.add(Dense(units=5, activation='relu', input_dim=784))
model.add(Dense(units=10, activation='softmax', input_dim= 5))
print(model.summary())

'''' 
-------------------------------------------------------------------------
Training
-------------------------------------------------------------------------
'''

# TODO complete the function compile here:
#model.compile()
opt = optimizers.Adam(lr = 0.01)
model.compile(loss='categorical_crossentropy', optimizer= opt, metrics=['accuracy'])


#Define the number of epochs and batch size:
n_epoch = 6
n_batch = 100

history = model.fit(X_train, Y_train, validation_data = (X_test,Y_test),  epochs=n_epoch, batch_size=n_batch, verbose = 1)


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


# Display some errors
pred_labels = np.argmax(predictions, axis=1)
idx_errors = np.where(pred_labels!=y_test[0:n_test])
plt.figure(figsize=(18,12))
K=0
#for i in range(12):
for i in idx_errors[0][0:12]:    
    plt.subplot(4,6,2*K+1)
    plot_image(i, predictions, y_test, X_test)
    plt.subplot(4,6,2*K+2)
    plot_value_array(i, predictions,  y_test)
    plt.xlabel(class_names[y_test[i]])
    K=K+1
plt.show()





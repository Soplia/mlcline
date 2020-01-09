from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import cluster, datasets
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model


'''' 
-------------------------------------------------------------------------
Dataset preparation
-------------------------------------------------------------------------
'''

n_samples = 4000
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.15)
noisy_blobs = datasets.make_blobs(n_samples=n_samples, cluster_std=0.3, centers=[(-0.1, -0.2), (1.2, 1.4)], n_features=2, random_state=0)
twocircles = datasets.make_circles(n_samples=n_samples, shuffle=True, noise=0.1, factor=0.3)
colors = np.array([x for x in 'bg'])
testcolors = np.array([x for x in 'mc'])

# Choose dataset here!
X, Y = noisy_blobs
#X, Y = noisy_moons 
#X, Y = twocircles 

# data type conversion
data = np.matrix(X).astype(np.float32)
Y = np.array(Y).astype(dtype=np.uint8)
labels = (np.arange(1) == Y[:, None]).astype(np.float32)

# split train and test :
X_train = data[0:3000,:]
Y_train = labels[0:3000,:]
X_test = data[3000:4000,:]
Y_test = labels[3000:,:]

# Display data
tr = plt.scatter(X[0:3000:, 0], X[0:3000, 1], marker='o', color=colors[Y[0:3000]].tolist(), s=10)
te = plt.scatter(X[3000:4000:, 0], X[3000:4000, 1], marker='x', color=colors[Y[3000:4000]].tolist(), s=10)
#plt.scatter(X[:, 0], X[:, 1], color=colors[Y].tolist(), s=10)
plt.legend((tr, te), ('training data', 'test data'), scatterpoints = 1, loc = 'lower right')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

''' 
-------------------------------------------------------------------------
Model definition
-------------------------------------------------------------------------
'''
model = Sequential()	
model.add(Dense(units=1, activation='sigmoid', input_dim=2))

print(model.summary())

''' 
-------------------------------------------------------------------------
Training the model
-------------------------------------------------------------------------
'''
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

history = model.fit(X_train, Y_train, shuffle = True, epochs=5, batch_size=1)
print(history.history.keys())

''' 
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

# Display the evolution of performance wrt epochs
plt.figure(figsize=(8,4))
# accuracy evaluation
plt.subplot(1,2,1)
plt.plot(history.history['acc'])
plt.title('Accuracy evolution')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')

plt.subplot(1,2,2)
# loss function evolution
plt.plot(history.history['loss'])
plt.title('Loss evolution')
plt.ylabel('Loss')
plt.xlabel('Epochs')

plt.show()

''' 
-------------------------------------------------------------------------
Visualization
-------------------------------------------------------------------------
'''

# display the decision border (y close to 0.5)
plt.figure()
# Generation of points in [-1,2]x[-1,2]
x_min = np.min(data[:,0])
x_max = np.max(data[:,0])
DATA_x = (np.random.rand(10**6,2)*(x_max-x_min))+x_min
# label prediction 
DATA_y = model.predict(DATA_x)
# display:
tr = plt.scatter(X[0:3000:, 0], X[0:3000, 1], marker='o', color=colors[Y[0:3000]].tolist(), s=10)
te = plt.scatter(X[3000:4000:, 0], X[3000:4000, 1], marker='x', color=colors[Y[3000:4000]].tolist(), s=10)
# display the decision border (y close to 0.5)
margin = 0.005
ind = np.where(np.logical_and(0.5-margin < DATA_y, DATA_y< 0.5+margin))[0]
DATA_ind = DATA_x[ind]
ss = plt.scatter(DATA_ind[:,0], DATA_ind[:,1], marker='_', color='red', s=2)
plt.legend((tr, te, ss), ('training data', 'test data', 'separation'), scatterpoints = 1, loc = 'lower right')
plt.xlabel('x1')
plt.ylabel('x2')
#plt.axis([-1,2,-1,2])
plt.show()


''' 
-------------------------------------------------------------------------
Saving the model
-------------------------------------------------------------------------
'''

#model.save_weights("model")
#plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=True)




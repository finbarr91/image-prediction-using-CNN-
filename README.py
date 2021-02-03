# image-prediction-using-CNN-
code in Python
# Importing the library
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,Dense, Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os


# Loading the dataset

(X_train,y_train), (X_test, y_test) = cifar10.load_data()
classes = ['Airplanes', 'Cars', ' Birds', 'Cats', 'Deer', 'Dogs', 'Frogs', 'Horses', 'Ships', 'Trucks']
print(X_train.shape)
print(X_test.shape)

# Visualizing our data
i = 2000
plt.imshow(X_train[i])
plt.title(y_train[i])
plt.show()

# Creating an image matrix
nrows = 15
ncols = 15

fig,axes = plt.subplots(nrows,ncols, figsize=(25,25))
axes = axes.ravel() # To flatten the 15x15 matrix to give 225
n_training = len(X_train) # 50000
for i in np.arange(0,nrows*ncols): # i ranges from 0 to 244

    index = np.random.randint(0,n_training) # pick a random number from the training data
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index],size=5 )
    axes[i].axis('off')
plt.tight_layout()
plt.subplots_adjust(hspace=1)
plt.show()

# Data Preparation
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

classes = 10
'''
Since we have 10 classes of the images, it is necessary we convert our y_train which originally has 0-9 
classes to be a categorically data(binary). This is perfect for the neural network
'''
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)
print('y_test\n', y_test)


# Performing Data Normalization: since the values of our pixel ranges from 0 to 255, we need to normalize
# it between 0 and 1
X_train = X_train/255
X_test = X_test/255
print('Normalized X_train\n', X_train)
print('Normalized X_test\n', X_test)

# To create an input shape since the previous input shape is (50000,32,32,3) so we want to remove the
# 50000 samples in other to feed this to the model. The goal is to extract the actual size of the images

input_shape = X_train.shape[1:]

cnn_model = Sequential()
cnn_model.add(Conv2D(filters=32, kernel_size= (3,3), activation='relu', input_shape=input_shape))
cnn_model.add(Conv2D(filters=32, kernel_size = (3,3), activation='relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.3))

cnn_model.add(Conv2D(filters=64, kernel_size= (3,3), activation='relu'))
cnn_model.add(Conv2D(filters= 64, kernel_size = (3,3), activation='relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.2))

cnn_model.add(Flatten())
cnn_model.add(Dense(units= 512, activation='relu'))
cnn_model.add(Dense(units= 512, activation= 'relu'))
cnn_model.add(Dense(units=10, activation = 'softmax'))

cnn_model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.rmsprop(lr=0.001), metrics=['acc'])
histroy = cnn_model.fit(X_train, y_train, batch_size= 32, epochs =10, shuffle =True)

# Evaluation of our model
'''
Remark:
In the evaluation of the model, the model is evaluated using the testing data
 '''
evaluation= cnn_model.evaluate(X_test,y_test)
print('Test Accuracy : {}'. format(evaluation[1]))

prediction= cnn_model.predict_classes(X_test)
print('Predicted classes of X_test\n', prediction)

'''
We need to return our y_test from binary to decimal(integer) values so as to make a good comparison of 
the predicted classes of x_test with the actual classes
'''
y_test = y_test.argmax(1)
print('y_test in decimal\n',y_test)

nrows = 7
ncols = 7

fig,axes = plt.subplots(nrows,ncols,figsize=(12,12))
axes = axes.ravel()

for i in np.arange(0, nrows*ncols):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction ={}\n True ={}'.format(prediction[i],y_test[i]), size=5)
    axes[i].axis('off')

plt.subplots_adjust(wspace=1, hspace=2)
plt.show()

# Confusion Matrix : This is used to summarize all our results in one matrix
cm = confusion_matrix(y_test, prediction)
print('Confusion Matrix\n', cm)
plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True)
plt.show()

# To save the model
directory = os.path.join(os.getcwd(), 'saved_models') # get the current directory and a file name called saved_model

if not os.path.isdir(directory): # If there is no folder called saved model in the directory, then create one
    os.makedirs(directory)

model_path = os.path.join(directory, 'keras_cifar10_trained_model.h5') # creating the model path
cnn_model.save(model_path) # saving the model path

# Training our model using the Augmented dataset

datagen = ImageDataGenerator(rotation_range= 90, width_shift_range= 0.1, horizontal_flip= True, vertical_flip= True)

datagen.fit(X_train)

# Trains the model on data generated batch-by-batch by a Python generator
cnn_model.fit_generator(datagen.flow(X_train,y_train, batch_size=32),epochs=2)

score = cnn_model.evaluate(X_test,y_test)
print('Test Accuracy: \n', score)

# save the model
directory = os.path.join(os.getcwd(), 'saved_models') # get the current directory and a file name called saved_model

if not os.path.isdir(directory): # If there is no folder called saved model in the directory, then create one
    os.makedirs(directory)

model_path = os.path.join(directory, 'keras_cifar10_trained_model.h5') # creating the model path
cnn_model.save(model_path) # saving the model path

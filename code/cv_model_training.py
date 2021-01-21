
#imports below
import tensorflow as tf
#from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

import pickle

pickle_in = open("X_train.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y_train.pickle","rb")
y = pickle.load(pickle_in)

y_binary = to_categorical(y)

X = X/255.0

x_train,x_val,y_train,y_val = train_test_split(X,y_binary,test_size=0.2)


datagen = ImageDataGenerator(
    rotation_range=90,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    horizontal_flip=True)
    #vertical_flip=True)
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
#datagen.fit(x_train)
NAME = "cv_model_training"
tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))
#model.add(Activation('relu'))

model.add(Dense(103))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# fits the model on batches with real-time data augmentation:
#model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
#                    steps_per_epoch=len(x_train) / 32, epochs=20, validation_data=(x_val,y_val), 
#                    callbacks=[tensorboard])

model.fit(datagen.flow(x_train, y_train), epochs=20, validation_data=(x_val,y_val),
        callbacks=[tensorboard])


model.save("cv_cnn_model.model")
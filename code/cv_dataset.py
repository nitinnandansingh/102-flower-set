import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import random
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataFile = "C:\\Users\\john\\Desktop\\Libin Docs\\Study Material\\WiSe2019-20\\Computer Vision\\Student Project\\jpg"
imageLabels = loadmat('imagelabels.mat')
classValue = imageLabels['labels']
labels = classValue[0]

path = os.path.join(dataFile)
image_size = 100
i = 0
training_data = []
data_temp = []

for img in tqdm(os.listdir(path)):  # iterate over each image
    try:
        class_value = labels[i]
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_UNCHANGED)# ,cv2.IMREAD_GRAYSCALE)  # convert to array
        RGB_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        new_array = cv2.resize(RGB_img, (image_size, image_size))  # resize to normalize data size
        training_data.append([new_array, class_value])  # add this to our training_data

        #flip_img = np.fliplr(new_array)
        #training_data.append([flip_img, class_value])  # add this to our training_data

        #rot_img = np.rot90(new_array)
        #training_data.append([rot_img, class_value])  # add this to our training_data

        i = i + 1
        #plt.imshow(new_array, cmap='gray')  # graph it
        #plt.show()  # display!
    except Exception as e:  # in the interest in keeping the output clean...
        pass
    #except OSError as e:
    #    print("OSErrroBad img most likely", e, os.path.join(path,img))
    #except Exception as e:
    #    print("general exception", e, os.path.join(path,img))


random.shuffle(training_data)
print("Done Shuffle")
X = []
y = []
x_temp = []
y_temp = []

for features,label in training_data:
    X.append(features)
    y.append(label)


print("done seperation")

X = np.array(X).reshape(-1, image_size, image_size, 3)
y = np.array(y)

print("Done Reshaping")

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

print("done splitting")

print(len(x_train))
print(len(x_test))

print(len(y_train))
print(len(y_test))


pickle_out = open("X_train.pickle","wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open("X_test.pickle","wb")
pickle.dump(x_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

print("DONE")
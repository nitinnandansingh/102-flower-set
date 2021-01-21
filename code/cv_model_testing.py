import cv2
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import average_precision_score

model = tf.keras.models.load_model("cv_cnn_model.model")

pickle_in = open("X_test.pickle","rb")
x = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y = pickle.load(pickle_in)

y_binary = to_categorical(y)

x = x/255.0

#y_pred = model.predict(x)
prediction = model.predict_classes(x,batch_size = 10)
print("Predicted Label:",prediction)                                                                
print("Actual Label:",y)
#for i,j in zip(prediction,y):
#    print(i,j)

cm = confusion_matrix(y,prediction)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#print(cm.diagonal())
sum = 0
count = 0
for i in cm.diagonal():
    sum = sum + i
    count = count + 1

average = sum / count

print("Accuracy Score:",accuracy_score(y, prediction)) # normalize=False))
print("Average Accuracy over Class Accuracy: ",average)
print("Total number of correct classification:",accuracy_score(y, prediction, normalize=False))
print("Total number of images in data set:", len(y))
#print(average_precision_score(y, prediction))
#average_precision = {}
#for i in range(1,103):
#    average_precision[i] = average_precision_score(y[:, i], prediction[:, i])
#average_precision["micro"] = average_precision_score(y, prediction, average="micro")
#print('Average precision score, micro-averaged over all classes: {0:0.2f}'
#      .format(average_precision["micro"]))
print("Precision Score:",precision_score(y, prediction, average="weighted", zero_division = 0))
#print(recall_score(y, prediction, average="weighted")) 
#print(cm)


plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


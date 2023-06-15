import os
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.layers import Conv2D, Flatten,Dense,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Adamax
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, classification_report


#Load Data
PresentWorkingDirectory = os.getcwd()
trainDataDirectory = PresentWorkingDirectory + "/data/train/"
trainingDataGenerator = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
train_gen = trainingDataGenerator.flow_from_directory(directory=trainDataDirectory, target_size=(48, 48), batch_size=64, color_mode="grayscale", class_mode="categorical", subset="training")

testDataDirectory = PresentWorkingDirectory + "/data/test/"
testingDataGenerator = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
test_gen  = testingDataGenerator.flow_from_directory(directory=testDataDirectory, target_size=(48, 48), batch_size=64, color_mode="grayscale", class_mode="categorical", subset="validation")

#Build Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

# Generate Pictorial Representation of the model
#model.summary()
#plot_model(model, show_layer_names=True)

model.compile( optimizer = Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
history = model.fit(x = train_gen, validation_data= test_gen, epochs=100)

#
#Plot Accuracy
#
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#
# Build Training Data Information
#
y_pred_train = model.predict(train_gen)
y_pred_formatted_train = numpy.argmax(y_pred_train, axis=1)
class_labels_train = train_gen.class_indices
class_labels_train = {v:k for k,v in class_labels_train.items()}
target_names_train = list(class_labels_train.values())

print(classification_report(train_gen.classes, y_pred_train, target_names=target_names_train))

confusionMatrix_train = confusion_matrix(train_gen.classes, y_pred_formatted_train)
disp_train = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix_train)
disp_train.plot()

tick_mark_train = numpy.arange(len(target_names_train))
_ = plt.xticks(tick_mark_train, target_names_train, rotation=90)
_ = plt.yticks(tick_mark_train, target_names_train)
plt.show()


#
# Build Validataion Data Information
#
y_pred_test = model.predict(test_gen)
y_pred_formatted = numpy.argmax(y_pred_test, axis=1)
class_labels = test_gen.class_indices
class_labels = {v:k for k,v in class_labels.items()}
target_names = list(class_labels.values())

print(classification_report(test_gen.classes, y_pred_test, target_names=target_names))

confusionMatrix = confusion_matrix(test_gen.classes, y_pred_formatted)
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix)
disp.plot()
tick_mark = numpy.arange(len(target_names))
_ = plt.xticks(tick_mark, target_names, rotation=90)
_ = plt.yticks(tick_mark, target_names)
plt.show()
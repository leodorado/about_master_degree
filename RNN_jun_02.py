import numpy as np
import pandas as pd
from numpy import array
import csv
import tensorflow
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras import optimizers
from tensorflow.keras import optimizers  #new line
from keras.layers import Conv2D,MaxPool2D,GlobalAveragePooling2D,AveragePooling2D
#from tensorflow.keras import GlobalAveragePooling2D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.metrics import classification_report,confusion_matrix

print('Implementación de Red Neuronal Recurrente     ')
print('   ')
# Data Reading 

Training_number_1 = 335   # 70% datos que NO tienen convulsiones
Training_number_2 = 85    # 70% datos que SI tienen convulsiones
list_as_array     = []
list_as_array2    = []

print('El 70% de los pacientes que NO tienen convulsiones es: ')
for i in range(1,476):
    if i<Training_number_1:
        with open('K%i.txt' %i, mode='r') as f:
            reader = csv.reader(f, delimiter='\t')
            data_as_list = list(reader)
        list_as_array.append(data_as_list)
        #print(np.shape(data_as_list))
    else:
        with open('K%i.txt' %i, mode='r') as f: #csv_file:
            reader = csv.reader(f, delimiter='\t')
            data_as_list = list(reader)
        #list_as_array2.append(reader)
        list_as_array2.append(data_as_list)
        #print(np.shape(data_as_list))

print('El 70% de los pacientes que SI tienen convulsiones es: ')
for i in range(1,120):
    if i<Training_number_2:
        with open('P%i.txt' %i, mode='r') as f: #csv_file:
            reader = csv.reader(f, delimiter='\t')
            data_as_list = list(reader)
        list_as_array.append(data_as_list)
        #print(np.shape(data_as_list))

    else:
        #print('el 30% es:')
        with open('P%i.txt' %i, mode='r') as f: #csv_file:
            reader = csv.reader(f, delimiter='\t')
            data_as_list = list(reader)
        #list_as_array2.append(reader)
        list_as_array2.append(data_as_list)
        #print(np.shape(data_as_list))
print('Finalización de ingreso de los datos...')
print('          ------------    ')

xtrain    = np.array(list_as_array, dtype=object).astype(float)
print('xtrain es: ')
print(np.shape(xtrain))
xtest     = np.array(list_as_array2).astype(float)
print(xtest.shape)

## Structure of the classification output
Train_number_N = 334;
Train_number_S = 84;
ytrain_1     = np.transpose(np.array([np.ones(Train_number_N),np.zeros(Train_number_N)]))
ytrain_a_1   = np.transpose(np.array([np.zeros(Train_number_S),np.ones(Train_number_S)]))
ytrain       = np.concatenate([ytrain_1,ytrain_a_1])
ytrain=np.array(ytrain)
print('ytrain es:', ytrain.shape)
#print(ytrain)

# Defining the sample number
Ns_N = 141;
Ns_S = 35;
ytest_1   = np.transpose(np.array([np.ones(Ns_N),np.zeros(Ns_N)]))
ytest_a_1 = np.transpose(np.array([np.zeros(Ns_S),np.ones(Ns_S)]))
ytest     = np.concatenate([ytest_1,ytest_a_1])
ytest     = np.array(ytest)
print('ytest es: ', ytest.shape)
#print(ytest)

#Neural Network Configuration

model=Sequential()
#model.add(GRU(32, input_shape=(np.array(xtrain).shape[1],7), return_sequences=False)) # For a GRU structure
#model.add(LSTM(64, input_shape=(np.array(xtrain).shape[1],3), return_sequences=False)) # For a LSTM structure
model.add(SimpleRNN(64, input_shape=(np.array(xtrain).shape[1],24), return_sequences=True)) # For a simple RNN
model.add(Dropout(0.2)) # best results of dropout
model.add(LSTM(64))
model.add(Dropout(0.2))

model.add(Dense(2, activation='sigmoid')) #best results for classification
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd=optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=None, decay=0.0)
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(model.summary())
print(np.array(xtrain).shape[1])

history=model.fit(xtrain, ytrain, epochs=10, batch_size=40, validation_data=(xtest, ytest))

score, acc = model.evaluate(xtest, ytest, batch_size=40)
print('Test score:', score)
print('Test accuracy:', acc)

scores = model.evaluate(xtest, ytest, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


Y_pred = model.predict(xtest)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)

incorrects = np.nonzero(model.predict_classes(xtest).reshape((-1,)) !=np.argmax(ytest, axis=1))
#print(model.predict_classes(xtest).reshape((-1,)))
#print(np.argmax(ytest, axis=1))
print(incorrects[0])

#Confussion Matrix

p=model.predict_step(xtest) # to predict probability

target_names = ['class 0(Gesture 1)', 'class 1(Gesture 2)']
print(classification_report(np.argmax(ytest,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(ytest,axis=1), y_pred))


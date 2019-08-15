import h5py
import numpy as np
#Anh Pham add this to ignore AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.utils import shuffle
from preprocessing.data_utils import get_ETL_data
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import *
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing.touchCount import touchCount
from preprocessing.touchCount import touchCountTensor
import time
from getPixel import printPixel
from keras.optimizers import Adam

model_path='weights/M11_touch_count.h5'
# max_records = 320 
max_records = 75 # Hiragana only
writers_per_char=160
chars, labs, spts=get_ETL_data('1' ,categories=range(0, max_records), writers_per_char=writers_per_char,database='ETL8B2',get_scripts=True)

unique_labels = list(set(labs))
unique_labels.sort()
labels_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
new_labels = np.array([labels_dict[l] for l in labs], dtype=np.int32)

x_get = chars.reshape(chars.shape[0],1,chars.shape[1],chars.shape[2])
y_get = np_utils.to_categorical(new_labels)

# Get only touchCount to be the input of model
start=time.time()
x1=np.array([touchCountTensor(x_get[i][0]) for i in range(len(x_get))])
end=time.time()
print('Time to run parameter-extractor: '+str(end-start))

xtc=x1.copy()
xtc=xtc.reshape(xtc.shape[0],1,xtc.shape[1],xtc.shape[2])
x_train,x_test,y_train,y_test=train_test_split(xtc,y_get,test_size=100,random_state=42)

model = Sequential()

model.add(Convolution2D(
    64, 3, strides=1, padding='same', input_shape=x_train.shape[1:], data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, strides=1, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(192, 3, strides=1, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(256, 3, strides=1, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(y_train.shape[1]))
model.add(Activation('softmax'))

try:
	model.load_weights(model_path)
except:
	pass
adam = Adam(lr=1e-4)
# model.compile(loss='categorical_crossentropy', optimizer=adam)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
start=time.time()
model.fit(x_train, y_train, epochs=10, shuffle=True)
end=time.time()
print('Time to fit model: '+str(end-start))

model.save(model_path)

# Evaluate model
y_pred=model.predict_classes(x_test)
y1=''
for i in range(len(y_test)):
	z=np.where(y_test[i]==1)[0][0]
	y1+=' '+str(z)
y1=y1.split(' ')
y1=y1[1:]
y1=[int(y1[i]) for i in range(len(y1))]
y1=np.asarray(y1)
accuracy_score(y_pred,y1)

# Print layers shape
def printModel(model):
	for i in range(len(model.layers)):
		print(str(i)+'.'+str(model.layers[i])+'. Input shape: '+str(model.layers[i].input_shape)+'. Output shape: '+str(model.layers[i].output_shape))


# 1st: 10 epochs - loss: 4.3245 - 0.4519 - acc: 0.0120 - 0.8573
# 2nd: 10 epochs - loss: 0.4043 - 0.1711 - acc: 0.8660 - 0.9367 - time: 1605
# 3rd: 10 epochs - loss: 0.1801 - 0.1074 - acc: 0.9361 - 0.9592 - time: 1631
# 4th: 10 epochs - loss: 0.1080 - 0.0667 - acc: 0.9591 - 0.9745 - time: 1605
# 5th: 10 epochs - loss: 0.0727 - 0.0602 - acc: 0.9722 - 0.9780 - time: 1540
# 6th: 10 epochs - loss: 0.0480 - 0.0445 - acc: 0.9823 - 0.9833 - time: 1524
# 7th: 10 epochs - loss: 0.0443 - 0.0416 - acc: 0.9843 - 0.9854 - time: 1639
# 8th: 10 epochs - loss: 0.0406 - 0.0323 - acc: 0.9846 - 0.9887 - time: 1660
# 9th: 10 epochs - loss: 0.0363 - 0.0310 - acc: 0.9880 - 0.9897 - time: 1560
#10th: 10 epochs - loss: 0.0284 - 0.0230 - acc: 0.9901 - 0.9921 - time: 1572
#11th:
#12th: 10 epochs - loss: 0.0175 - 0.0286 - acc: 0.9929 - 0.9915 - time: 1654
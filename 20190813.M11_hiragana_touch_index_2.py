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
from preprocessing.touchCount import touchIndexTensor2
import time
from getPixel import printPixel
from keras.optimizers import Adam

model_path='weights/M11_touch_index_2.h5'
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

# Get only touchIndex to be the input of model
start=time.time()
x1=np.array([touchIndexTensor2(x_get[i][0]) for i in range(len(x_get))])
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


# 1st: 10 epochs - loss: 4.3231 - 0.4154 - acc: 0.0139 - 0.8637 - time: 1812
# 2nd: 10 epochs - loss: 0.3672 - 0.1748 - acc: 0.8777 - 0.9371 - time: 1868
# 3rd: 10 epochs - loss: 0.3417 - 0.0954 - acc: 0.8885 - 0.9651 - time: 1889
# 4th: 10 epochs - loss: 0.0917 - 0.0603 - acc: 0.9649 - 0.9782 - time: 1869
# 5th: 10 epochs - loss: 0.0567 - 0.0562 - acc: 0.9782 - 0.9794 - time: 1850
# 6th: 10 epochs - loss: 0.0489 - 0.0338 - acc: 0.9825 - 0.9871 - time: 1854
# 7th: 10 epochs - loss: 0.0391 - 0.0420 - acc: 0.9858 - 0.9864 - time: 1861
# 8th: 10 epochs - loss: 0.0412 - 0.0279 - acc: 0.9861 - 0.9895 - time: 1829
# 8th: 10 epochs - loss: 0.0311 - 0.0298 - acc: 0.9903 - 0.9903 - time: 1797
# 9th: 10 epochs - loss: 0.0251 - 0.0207 - acc: 0.9918 - 0.9923 - time: 1781
#10th: 10 epochs - loss: 0.0213 - 0.0179 - acc: 0.9936 - 0.9944 - time: 1773
#11th: 10 epochs - loss: 0.0114 - 0.0131 - acc: 0.9957 - 0.9961 - time: 1876
#12th: 10 epochs - loss: 0.0196 - 0.0180 - acc: 0.9935 - 0.9953 - time: 1811
#13th: 10 epochs - loss: 0.0138 - 0.0270 - acc: 0.9955 - 0.9932 - time: 1846
#14th: 10 epochs - loss: 0.0163 - 0.0127 - acc: 0.9945 - 0.9962 - time: 1761
import h5py
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from preprocessing.make_keras_input import data
from dictionaryHiragana import LookUp
from tensorflow import newaxis
from PIL import Image
from preprocessing.touchCount import touchCountTensor
from keras.models import load_model

model_path='weights/M11_touch_count.h5'

n_output = 75
model = load_model(model_path)
path='chugiday.png'
im=Image.open(path)
im=im.resize((64,64))
im=im.convert(mode='L')
data=np.asarray(im)
data=data/255
data=touchCountTensor(data)
data=data[newaxis,newaxis,...]
result=model.predict_classes(data)
formatResult=LookUp(result[0])
print('Result: Character \''+formatResult+'\' - Hiragana')

# bChoice=input('Is this result truthy (y/n)?')
# if bChoice.lower()=='n':
# 	y_train=input('I will retrain the model. Enter label for this sample: ')
# 	x_train=data
# 	reTrain_M7(model,x_train,y_train)
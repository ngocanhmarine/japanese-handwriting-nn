import h5py
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from preprocessing.make_keras_input import data
from dictionaryHiragana import LookUp
from tensorflow import newaxis
from PIL import Image
from models import M7_1
def load_model_weights(name, model):
    try:
        model.load_weights('weights/M7_1-hiragana_weights.h5')
    except:
        print("Can't load weights!")


def save_model_weights(name, model):
    try:
        model.save_weights(name)
    except:
        print("failed to save classifier weights")
    pass

n_output = 75
model = M7_1(n_output=n_output, input_shape=(1, 64, 64))
load_model_weights('weights/M7_1-hiragana_weights.h5', model)
path='chugiday.png'
im=Image.open(path)
im=im.resize((64,64))
im=im.convert(mode='L')
data=np.asarray(im)
data=data/255
data=data[newaxis,newaxis,...]
result=model.predict_classes(data)
print('Result: Character \''+LookUp(result[0])+'\' - Hiragana')

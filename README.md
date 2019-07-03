# japanese-handwriting-nn

This is updated version to: https://github.com/charlietsai/japanese-handwriting-nn

Handwritten japanese character recognition using neural networks.

Read the corresponding paper [here](writeup.pdf).

An example job running the M16 model on the hiragana dataset is included [here](example_job.py). 

You will need to obtain the ETL Character Database [here](http://etlcdb.db.aist.go.jp/) and make sure the `ETL_path` in [`/preprocessing/data_utils.py`](/preprocessing/data_utils.py) is correct.
* Updated contents:
- Updated to run in Python3.
- Backend compability with both Theano and Tensorflow.
- Add modules predict-retrain.
- Example with model M7, M11, M16
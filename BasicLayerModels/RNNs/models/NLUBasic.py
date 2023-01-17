from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import *
import tensorflow as tf
import os

class NLUModel:
    def __init__(self):
        self.model = None

    def predict(self,x):
        return self.model.predict(x)

    def save(self,model_path,model_name):
        self.model.save(os.path.join(model_path,'{}.h5'.format(model_name)))

    def save_weights(self, model_path, model_name):
        self.model.save_weights(os.path.join(model_path, '{}.h5'.format(model_name)))

    def load_weights(self, model_path, model_name):
        self.model.load_weights(os.path.join(model_path, '{}.h5'.format(model_name)))

    def load(self,model_path,custom_objects=None):
        self.model = load_model(model_path,custom_objects=custom_objects)

    def fit_val(self,X,Y,valid_data=None,epoch=5,batch_size=32):
        self.history = self.model.fit(X, Y, validation_data=valid_data, epochs=epoch, batch_size=batch_size)

    def fit_train(self, X, Y, val_split=0.1, epoch=5, batch_size=32):
        self.history = self.model.fit(X, Y, validation_split=val_split, epochs=epoch, batch_size=batch_size)

    def get_opts(self,opt_names,lr):
        if opt_names == 'Adam':
            optim = Adam(learning_rate=lr)
        elif opt_names == 'Adadelta':
            optim = Adadelta(learning_rate=lr)
        elif opt_names == 'Adagrad':
            optim = Adagrad(learning_rate=lr)
        elif opt_names == 'RMSProp':
            optim = RMSprop(learning_rate=lr)
        elif opt_names == 'SGD':
            optim = SGD(learning_rate=lr)
        else:
            optim = SGD(learning_rate=lr)
        return optim

    def build_model(self):
        raise NotImplementedError

    def compile_model(self):
        raise NotImplementedError

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import *
import tensorflow as tf
class NLUModel:
    def __init__(self):
        self.model = None

    def predict(self,x):
        return self.model.predict(x)

    def save(self,model_path):
        self.model.save(model_path)

    def load(self,model_path,custom_objects=None):
        self.model = load_model(model_path,custom_objects=custom_objects)
        
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
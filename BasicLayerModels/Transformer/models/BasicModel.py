import os

class BasicModel:
    def __init__(self):
        self.model = None

    def predict(self,x):
        assert self.model is not None, "model object is None!"
        return self.model.predict(x)

    def save(self,save_path,model_name):
        assert self.model is not None, "model object is None!"
        self.model.save(os.path.join(save_path,model_name))

    def fit(self,X,Y,epochs,batch_size,valid_data=None):
        raise NotImplementedError


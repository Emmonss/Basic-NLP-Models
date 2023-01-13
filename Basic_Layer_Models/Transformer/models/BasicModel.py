import os

class BasicModel:
    def __init__(self):
        self.model = None

    def predict(self,x):
        # print(type(self.model))
        # assert self.model == None, "model is None"
        return self.model.predict(x)

    def save(self,save_path,model_name):
        self.model.save(os.path.join(save_path,model_name))
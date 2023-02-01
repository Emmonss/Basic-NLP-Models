
import os,json
import tensorflow as tf
from BasicLayerModels.Transformer.backend.DictToClass import DictToClass
from BasicLayerModels.Transformer.models.BasicModel import BasicModel
from BasicLayerModels.Transformer.models.load_model import load_bert_from_ckpt
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense,Dropout
class BertModelForSimsent(BasicModel):
    def __init__(self,config,**kwargs):
        '''
        :param config:
                needed: config_path
                        ckpt_path
                        dropout
                        class_num
                        lr
                    #train val predict
                        epochs
                        batch_size
        '''
        super(BertModelForSimsent, self).__init__()
        self.config = config
        self.build(**kwargs)
        self.compile_model()

    def build(self,**kwargs):
        bert_model =load_bert_from_ckpt(self.config.config_path,
                                        self.config.ckpt_path,**kwargs)

        if bert_model is None:
            raise ValueError("bert_model is None!")
        seq, seg = bert_model.input

        bert_out = bert_model.output
        bert_sent = bert_out[:, 0, :]
        bert_sent_drop = Dropout(rate=self.config.dropout, name="bert_sent_drop")(bert_sent)

        sent_tc = Dense(self.config.class_num, activation='softmax', name='sim_classifier')(bert_sent_drop)
        self.model = Model(inputs=[seq, seg], outputs=[sent_tc])
        self.model.summary()

    def fit(self,X,Y,valid_data=None,epochs=6,batch_size=32):
        if self.model is None:
            raise ValueError("model is None")
        self.model.fit(X, Y, validation_data=valid_data, epochs=epochs, batch_size=batch_size)

    def compile_model(self):
        opt = tf.keras.optimizers.Adam(lr=self.config.lr)
        loss = {
            'sim_classifier':'sparse_categorical_crossentropy'
        }
        loss_weight = {'sim_classifier':1.0}
        metrics = {'sim_classifier':'acc'}
        self.model.compile(optimizer=opt,loss=loss,metrics=metrics,loss_weights=loss_weight)


if __name__ == '__main__':
    config_dict = {
        "max_len" : 128,
        "do_lower_case" : True,
        "dropout" : 0.1,
        "class_num" : 2,
        "epoch" : 5,
        "batch_size" : 16,
        "lr" : 1e-5,
        "vocab_path" : '../../modelHub/chinese_L-12_H-768_A-12/vocab.txt',
        "config_path" : '../../modelHub/chinese_L-12_H-768_A-12/bert_config.json',
        "ckpt_path" : '../../modelHub/chinese_L-12_H-768_A-12/bert_model.ckpt'
    }
    config = DictToClass(**config_dict)
    model = BertModelForSimsent(config=config)
    pass


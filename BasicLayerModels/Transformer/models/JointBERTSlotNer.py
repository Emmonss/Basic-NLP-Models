from BasicLayerModels.Transformer.models.BasicModel import BasicModel
from BasicLayerModels.Transformer.bertModels.LoadModel import load_bert_from_ckpt
from tensorflow.python.keras.layers import Dense,Dropout
from tensorflow.python.keras.models import Model
from BasicLayerModels.Transformer.backend.snippets import DictToClass,ClassToDict,read_json_1dict,write_json_2format_1dict
from datetime import datetime
import tensorflow as tf
import os


class JointBertSlotNer(BasicModel):
    def __init__(self,intent_num,slot_num,config,**kwargs):
        '''
        :param config:
            need:   slot_num
                    intent_num
                    config_path
                    ckpt_path
                    dropout
                    lr
                #train val predict
                    epoch
                    batch_size
        '''
        super(JointBertSlotNer, self).__init__()
        self.intent_num=intent_num
        self.slot_num = slot_num
        self.config = config
        self.build_model(**kwargs)
        self.compile_model()


    def build_model(self,**kwargs):
        bert_model = load_bert_from_ckpt(self.config.config_path,
                                        self.config.ckpt_path,**kwargs)
        if bert_model is None:
            raise ValueError("bert_model is None!")
        seq, seg = bert_model.input

        bert_seq_output = bert_model.output
        bert_pooled_output = bert_seq_output[:,0,:]
        intent_drop = Dropout(rate=self.config.dropout,name='intent_drop')(bert_pooled_output)
        intent_fc = Dense(self.intent_num,activation='softmax',name="intent_classifier")(intent_drop)

        slot_drop = Dropout(rate=self.config.dropout,name='slot_drop')(bert_seq_output)
        slot_fc = Dense(self.slot_num,activation='softmax',name="slot_classifier")(slot_drop)

        self.model = Model(inputs=[seq, seg],outputs=[slot_fc,intent_fc])
        self.model.summary()

    def compile_model(self):
        opt = tf.keras.optimizers.Adam(lr=self.config.lr)
        loss = {
            'slot_classifier': 'sparse_categorical_crossentropy',
            'intent_classifier': 'sparse_categorical_crossentropy'
        }
        loss_weight = {'slot_classifier':3.0,'intent_classifier':1.0}
        metrics = {'slot_classifier': 'acc', 'intent_classifier': 'acc'}
        self.model.compile(optimizer=opt,loss=loss,metrics=metrics,loss_weights=loss_weight)

    def fit(self,X,Y,epochs=1,batch_size=16,valid_data=None):
        if self.model is None:
            raise ValueError("model is None")
        self.model.fit(X, Y, validation_data=valid_data, epochs=epochs, batch_size=batch_size)

    def save(self, save_path, model_name):
        time_date = datetime.now()
        time = '{}-{}-{}-{}'.format(time_date.year, time_date.month, time_date.day, time_date.hour)
        if not os.path.exists(os.path.join(save_path, time)):
            os.makedirs(os.path.join(save_path, time))
        assert self.model is not None, "model object is None!"
        self.model.save(os.path.join(save_path, time,
                                     '{}.h5'.format(model_name)))

        write_json_2format_1dict(ClassToDict(self.config),
                                 os.path.join(save_path, time, '{}.json'.format(model_name)))


if __name__ == '__main__':
    config_dict = {
        "intent_num":10,
        "slot_num":20,
        "max_len": 128,
        "do_lower_case": True,
        "dropout": 0.1,
        "class_num": 2,
        "epoch": 5,
        "batch_size": 16,
        "lr": 1e-5,
        "vocab_path": '../../modelHub/chinese_L-12_H-768_A-12/vocab.txt',
        "config_path": '../../modelHub/chinese_L-12_H-768_A-12/bert_config.json',
        "ckpt_path": '../../modelHub/chinese_L-12_H-768_A-12/bert_model.ckpt'
    }
    config = DictToClass(**config_dict)
    model = JointBertSlotNer(config=config,
                             intent_num=config.intent_num,
                             slot_num=config.slot_num)
    pass






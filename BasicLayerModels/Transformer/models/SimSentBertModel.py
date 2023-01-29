
import os,json
import tensorflow as tf

from BasicLayerModels.RNNs.models.NLUBasic import NLUModel
from BasicLayerModels.Transformer.models.BERT import BERT
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense,Dropout

class BertModelForSimsent(NLUModel):
    def __init__(self,config):
        super(BertModelForSimsent, self).__init__()
        self.config = config
        self.load_bert()
        self.build()
        self.compile_model()

    def load_bert(self,**kwargs):
        config_path = self.config["config_path"]
        checkpoint_path =self.config["ckpt_path"]
        configs = {}
        if config_path is not None:
            configs.update(json.load(open(config_path)))
        configs.update(kwargs)

        if 'max_position' not in configs:
            configs['max_position'] = configs.get('max_position_embeddings', 512)
        if 'dropout_rate' not in configs:
            configs['dropout_rate'] = configs.get('hidden_dropout_prob')
        if 'segment_vocab_size' not in configs:
            configs['segment_vocab_size'] = configs.get('type_vocab_size', 2)

        bert_model = BERT(**configs)
        bert_model.build()

        if checkpoint_path is not None:
            bert_model.load_weight_from_checkpoint(checkpoint_path)
        return bert_model.model

    def build(self):
        bert_model = self.load_bert()

        if bert_model is None:
            raise ValueError("bert_model is None!")
        seq, seg = bert_model.input

        bert_out = bert_model.output
        bert_sent = bert_out[:, 0, :]
        bert_sent_drop = Dropout(rate=self.config["dropout"], name="bert_sent_drop")(bert_sent)

        sent_tc = Dense(self.config["class_num"], activation='softmax', name='sim_classifier')(bert_sent_drop)
        self.model = Model(inputs=[seq, seg], outputs=[sent_tc])
        self.model.summary()

    def fit(self,X,Y,valid_data=None,epochs=6,batch_size=32):
        if self.model is None:
            raise ValueError("model is None")
        self.model.fit(X, Y, validation_data=valid_data, epochs=epochs, batch_size=batch_size)

    def compile_model(self):
        opt = tf.keras.optimizers.Adam(lr=self.config["learning_rate"])
        loss = {
            'sim_classifier':'sparse_categorical_crossentropy'
        }
        loss_weight = {'sim_classifier':1.0}
        metrics = {'sim_classifier':'acc'}
        self.model.compile(optimizer=opt,loss=loss,metrics=metrics,loss_weights=loss_weight)


if __name__ == '__main__':
    config123 = {
        "config_path" : '../../modelHub/chinese_L-12_H-768_A-12/bert_config.json',
        "ckpt_path" : '../../modelHub/chinese_L-12_H-768_A-12/bert_model.ckpt',
        "dropout":0.1,
        "class_num":2,
        "learning_rate":1e-5
    }
    bs = BertModelForSimsent(config123)
    print(type(bs.model))
    bs.save("./","test")


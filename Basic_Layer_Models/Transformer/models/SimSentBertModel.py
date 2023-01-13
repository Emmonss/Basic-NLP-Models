
import os,json
import tensorflow as tf
from Basic_Layer_Models.Transformer.models.BERT import BERT
from Basic_Layer_Models.Transformer.models.BasicModel import BasicModel
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense,Dropout

class BertModel_for_Simsent(BasicModel):
    def __init__(self,config):
        super(BertModel_for_Simsent,self).__init__()
        self.config = config
        self.load_bert()
        self.build()

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
        # self.model = bert_model.model

    def build(self,**kwargs):
        bert_model = self.load_bert()

        if bert_model is None:
            raise ValueError("bert_model is None!")
        seq, seg = bert_model.input

        bert_out = bert_model.output
        bert_sent = bert_out[:, 0, :]
        bert_sent_drop = Dropout(rate=config["dropout"], name="bert_sent_drop")(bert_sent)

        sent_tc = Dense(config["class_num"], activation='softmax', name='sim_classifier')(bert_sent_drop)
        self.model = Model(inputs=[seq, seg], outputs=[sent_tc])
        self.model.summary()
        self.model.save(os.path.join("./","test.h5"))


if __name__ == '__main__':
    config = {
        "config_path" : '../../model_hub/chinese_L-12_H-768_A-12/bert_config.json',
        "ckpt_path" : '../../model_hub/chinese_L-12_H-768_A-12/bert_model.ckpt',
        "dropout":0.1,
        "class_num":2
    }
    bert = BertModel_for_Simsent(config)
    print(type(bert.model))
    bert.save("./","test.h5")


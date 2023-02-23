import json

from BasicLayerModels.Transformer.bertModels.BERT import BERT
from BasicLayerModels.Transformer.bertModels.ALBERT import ALBERT
from BasicLayerModels.Transformer.backend.snippets import *
def load_bert(config_path,
              checkpoint_path,
              model_type ='bert',
              **kwargs):
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)

    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings',512)
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size',2)

    models={
        'bert':BERT,
        'albert':ALBERT
    }
    if is_string(model_type):
        model_type = model_type.lower()
        if model_type in models.keys():
            MODEL = models.get(model_type)
        else:
            MODEL=BERT
    else:
        MODEL = model_type

    bert_model = MODEL(**configs)
    bert_model.build(**configs)

    if checkpoint_path is not None:
        bert_model.load_weight_from_checkpoint(checkpoint_path)

    return bert_model.model

if __name__ == '__main__':
    pass
    #BERT
    config_path = 'modelHub/chinese_L-12_H-768_A-12/bert_config.json'
    ckpt_path = 'modelHub/chinese_L-12_H-768_A-12/bert_model.ckpt'
    model = load_bert(config_path=config_path,
                      checkpoint_path=ckpt_path,
                      model_type='BERT')
    print(model.summary())

    config_path = 'modelHub/albert/albert_config.json'
    ckpt_path = 'modelHub/albert/albert_model.ckpt'
    model = load_bert(config_path=config_path,
                      checkpoint_path=ckpt_path,
                      model_type='albert')
    print(model.summary())

    # from bert4keras.models import build_transformer_model
    #
    # models = build_transformer_model(config_path=config_path,checkpoint_path = ckpt_path,models='bert')
    # models.summary()

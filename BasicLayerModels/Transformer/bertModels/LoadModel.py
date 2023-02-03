import json
from BasicLayerModels.Transformer.bertModels.BERT import BERT

def load_bert_from_ckpt(config_path,ckpt_path,**kwargs):
    config_path = config_path
    checkpoint_path = ckpt_path
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
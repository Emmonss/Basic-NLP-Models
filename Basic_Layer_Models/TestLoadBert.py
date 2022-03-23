import json

from Basic_Layer_Models.Transformer.models.BERT import BERT


def load_bert(cofig_path,
              checkpoint_path,
              **kwargs):
    cofigs = {}
    if cofig_path is not None:
        cofigs.update(json.load(open(cofig_path)))
    cofigs.update(kwargs)

    if 'max_position' not in cofigs:
        cofigs['max_position'] = cofigs.get('max_position_embeddings',512)
    if 'dropout_rate' not in cofigs:
        cofigs['dropout_rate'] = cofigs.get('hidden_dropout_prob')
    if 'segment_vocab_size' not in cofigs:
        cofigs['segment_vocab_size'] = cofigs.get('type_vocab_size',2)

    bert_model = BERT(**cofigs)
    bert_model.build()

    if checkpoint_path is not None:
        bert_model.load_weight_from_checkpoint(checkpoint_path)
    print(bert_model.model.summary())
    return bert_model.model

if __name__ == '__main__':
    config_path = '../model_hub/chinese_L-12_H-768_A-12/bert_config.json'
    ckpt_path = '../model_hub/chinese_L-12_H-768_A-12/bert_model.ckpt'

    model = load_bert(cofig_path=config_path,checkpoint_path=ckpt_path)
    # from bert4keras.models import build_transformer_model

    # model = build_transformer_model(config_path=config_path,checkpoint_path = ckpt_path,model='bert')
    # model.summary()


#basic
max_len=128
do_lower_case=True
dropout=0.1
class_num=2
epoch=5
batch_size=16
lr = 1e-5
save_path = './result'

#BERT
bert_vocab_path = '../../BasicLayerModels/modelHub/chinese_L-12_H-768_A-12/vocab.txt'
bert_config_path = '../../BasicLayerModels/modelHub/chinese_L-12_H-768_A-12/bert_config.json'
bert_ckpt_path = '../../BasicLayerModels/modelHub/chinese_L-12_H-768_A-12/bert_model.ckpt'
bert_model_type = 'bert'

#ALBERT
albert_vocab_path = '../../BasicLayerModels/modelHub/albert/vocab.txt'
albert_config_path = '../../BasicLayerModels/modelHub/albert/albert_config.json'
albert_ckpt_path = '../../BasicLayerModels/modelHub/albert/albert_model.ckpt'
albert_model_type = 'albert'

#main
vocab_path = albert_vocab_path
config_path = albert_config_path
ckpt_path = albert_ckpt_path
model_type = albert_model_type

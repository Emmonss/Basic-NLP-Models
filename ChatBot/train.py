
import os,json
import numpy as np
from ChatBot.sentToken import DialogCodeTrans
from Basic_Layer_Models.RNNs.models.SequenceAttention import Seq2SeqAttention
from ChatBot import train_config
from datetime import datetime
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
    except RuntimeError as e:
        print(e)



def train_di():
    #init
    inpath = train_config.inpath_di
    tarpath = train_config.tarpath_di
    save_path = train_config.save_model_path_di
    model_name = train_config.model_name_di
    dict_path = train_config.dict_vocab_pure

    #
    tokenizer = DialogCodeTrans(dict_path=dict_path,
                                sent_maxlen=train_config.sent_maxlen,
                                pre_token_flag=train_config.pretoken_flg)

    print("loading dialog and encoding")
    token_input_list, token_target_list = tokenizer.sent_2_idx(inpath, tarpath)
    print("input token shape:{}".format(np.shape(token_input_list)))
    print("target token shape:{}".format(np.shape(token_target_list)))

    model = Seq2SeqAttention(vocab_size=tokenizer.vocab_size,
                             embeding_dim=train_config.embed_dim,
                             max_sent_len=train_config.sent_maxlen,
                             encoder_units=train_config.hidden_units,
                             lr=train_config.lr,
                             teach_forcing_ran=train_config.teach_forcing_ran
                             )
    print("training......")
    model.fit_val(X=[token_input_list,token_target_list],
                  Y=token_target_list[:,1:],
                  epoch=train_config.epoch,
                  batch_size=train_config.batch_size)
    print("done!")

    if train_config.saving_flg:
        save_path = save_path.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_weights(model_path=save_path, model_name=model_name)
        basic_params = {
            'vocab_size':tokenizer.vocab_size,
            'embed_dim':train_config.embed_dim,
            'sent_maxlen':train_config.sent_maxlen,
            'hidden_units':train_config.hidden_units,
            'lr':train_config.lr,
            'epoch': train_config.epoch,
            'batch_size' : train_config.batch_size,
            'pretoken_flg':train_config.pretoken_flg,
            'dict_vocab_pure':dict_path,
            'inpath' : inpath,
            'tarpath' : tarpath
        }
        with open('{}/param.json'.format(save_path),'w',encoding='utf-8') as fw:
            json.dump(basic_params,fw,ensure_ascii=False)


def train_xhj():
    inpath = train_config.inpath_xhj
    tarpath = train_config.tarpath_xhj
    save_path = train_config.save_model_path_xhj
    model_name = train_config.model_name_xhj
    dict_path = train_config.dict_vocab_pure

    tokenizer = DialogCodeTrans(dict_path=dict_path,
                                sent_maxlen=train_config.sent_maxlen,
                                pre_token_flag=train_config.pretoken_flg)
    #
    print("loading dialog and encoding")
    token_input_list, token_target_list = tokenizer.sent_2_idx(inpath, tarpath)
    print("input token shape:{}".format(np.shape(token_input_list)))
    print("target token shape:{}".format(np.shape(token_target_list)))

    model = Seq2SeqAttention(vocab_size=tokenizer.vocab_size,
                             embeding_dim=train_config.embed_dim,
                             max_sent_len=train_config.sent_maxlen,
                             encoder_units=train_config.hidden_units,
                             lr=train_config.lr,
                             teach_forcing_ran=train_config.teach_forcing_ran
                             )
    print("training......")
    model.fit_val(X=[token_input_list, token_target_list],
                  Y=token_target_list[:, 1:],
                  epoch=train_config.epoch,
                  batch_size=train_config.batch_size)
    print("done!")

    if train_config.saving_flg:
        save_path = save_path.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_weights(model_path=save_path, model_name=model_name)
        basic_params = {
            'vocab_size': tokenizer.vocab_size,
            'embed_dim': train_config.embed_dim,
            'sent_maxlen': train_config.sent_maxlen,
            'hidden_units': train_config.hidden_units,
            'lr': train_config.lr,
            'epoch': train_config.epoch,
            'batch_size': train_config.batch_size,
            'pretoken_flg': train_config.pretoken_flg,
            'dict_vocab_pure': dict_path,
            'inpath': inpath,
            'tarpath': tarpath
        }
        with open('{}/{}'.format(save_path,train_config.params_save_name), 'w', encoding='utf-8') as fw:
            json.dump(basic_params, fw, ensure_ascii=False)

if __name__ == '__main__':
    train_xhj()
    pass